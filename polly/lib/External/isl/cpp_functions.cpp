#include "isl/aff.h"
#include "isl/ctx.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/schedule_node.h"
#include "isl/schedule_type.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include <iostream>
#include <limits>
#include <vector>

#include "cpp_functions.h"
#include "polly/Support/ISLTools.h"

isl::stat isl_id_to_id_foreach(
	isl_id_to_id *id_to_id, std::function<isl::stat(isl::id, isl::id)> f)
{
	struct Data {
		decltype(f) &fn;
	} data{f};
	auto lambda2 = [](isl_id *id, isl_id *target_id, void *user) -> isl_stat {
		return static_cast<struct Data *>(user)
			->fn(isl::manage(id), isl::manage(target_id))
			.release();
	};
	return isl::manage(isl_id_to_id_foreach(id_to_id, lambda2, &data));
}

/// Tag the @p Relation domain with @p TagId
static __isl_give isl_map *tag(
	__isl_take isl_map *Relation, __isl_take isl_id *TagId)
{
	isl_space *Space = isl_map_get_space(Relation);
	Space = isl_space_drop_dims(
		Space, isl_dim_out, 0, isl_map_dim(Relation, isl_dim_out));
	Space = isl_space_set_tuple_id(Space, isl_dim_out, TagId);
	isl_multi_aff *Tag = isl_multi_aff_domain_map(Space);
	Relation = isl_map_preimage_domain_multi_aff(Relation, Tag);
	return Relation;
}

isl::union_map tag(isl::union_map umap, isl::id id)
{
	isl::union_map taggedMap =
		isl::manage(isl_union_map_empty(umap.get_space().release()));
	umap.foreach_map([&](isl::map map) {
		isl::map tagged = isl::manage(tag(map.release(), id.copy()));
		taggedMap = taggedMap.unite(tagged.to_union_map());
		return isl::stat::ok();
	});
	return taggedMap;
}

isl_union_map *isl_ast_generate_array_expansion_indexing(
	isl_schedule_node *_node, isl_union_map *_extra_umap, isl_union_map *_lrs)
{
	ISL_DEBUG(std::cerr << "isl_ast_generate_array_expansion_indexing\n");
	isl::union_map lrs = isl::manage(_lrs);
	ISL_DEBUG(std::cerr << "lrs "; isl_union_map_dump(lrs.get()));

	isl::schedule_node_band node =
		isl::manage_copy(_node).as<isl::schedule_node_band>();
	isl::union_map extra_umap = isl::manage(_extra_umap);
	isl_ctx *ctx = node.ctx().get();

	auto r = node.n_member();
	if (r.is_error())
		return nullptr;
	int n_members = (unsigned)r;

	r = node.get_schedule_depth();
	if (r.is_error())
		return nullptr;
	int depth = (unsigned)r;

	// TODO
	int n_expanded_dims = depth + n_members;

	isl::union_map full_array_umap;

	for (int member = 0; member < n_members; member++) {

		isl_id_to_id *array_expansion =
			isl_schedule_node_band_member_get_array_expansion(
				node.get(), member);

		auto partial_schedule = node.get_partial_schedule().at(member);
		ISL_DEBUG(std::cerr << "partial_schedule ";
				  isl_union_pw_aff_dump(partial_schedule.get()));

		isl::union_map member_array_umap;

		auto lambda1 = [&](isl::id array_id, isl::id target_id) -> isl::stat {
			int factor = (int)(uintptr_t)target_id.get_user();
			isl::id offset_base_stmt;

			isl::union_map tagged_partial_schedule =
				tag(partial_schedule.as_union_map(), array_id);
			ISL_DEBUG(std::cerr << "tagged_partial_schedule ";
					  isl_union_map_dump(tagged_partial_schedule.get()));

			isl::union_map umap;
			extra_umap.foreach_map([&](isl::map map) {
				auto r = map.dim(isl::dim::param);
				if (r.is_error())
					return isl::stat::error();
				int n_param = (unsigned)r;
				r = map.get_space().dim(isl::dim::out);
				if (r.is_error())
					return isl::stat::error();
				int n_stmt_dims = (unsigned)r;
				n_stmt_dims = 0;
				auto stmt_id = map.get_range_tuple_id();

				ISL_DEBUG(std::cerr << "handling "; isl_id_dump(stmt_id.get()));

				int offset = [&]() {
					if (factor == 1)
						return 0;

					// FIXME This assumes there is no disjoint sets of live
					// range spans wrt to a specific array in each band. I.e.
					// for all statements that access the same array that got
					// expanded, we will always be able to get a valid offset
					// for all statements wrt an arbitrary single one.
					if (offset_base_stmt.is_null()) {
						offset_base_stmt = stmt_id;
					}
					ISL_DEBUG(std::cerr << "offset_base_stmt: ";
							  isl_id_dump(offset_base_stmt.get()));

					int invalid_offset = std::numeric_limits<int>::min();
					int offset = invalid_offset;

					bool lrs_relation_found = false;

					bool is_intra_relation =
						stmt_id.get() == offset_base_stmt.get();

					lrs.foreach_map([&](isl::map map) {
						// map is in a space ((S1->A) -> (S2->A))
						isl::id domain_stmt = map.get_space()
												  .domain()
												  .unwrap()
												  .get_domain_tuple_id();
						isl::id range_stmt = map.get_space()
												 .range()
												 .unwrap()
												 .get_domain_tuple_id();
						isl::id map_array_id = map.get_space()
												   .domain()
												   .unwrap()
												   .get_range_tuple_id();
						[[maybe_unused]] isl::id map_range_array_id =
							map.get_space()
								.range()
								.unwrap()
								.get_range_tuple_id();
						isl_assert(ctx,
							map_array_id.get() == map_range_array_id.get(),
							abort());

						if (array_id.get() != map_array_id.get())
							return isl::stat::ok();

						bool is_positive_offset;
						if (domain_stmt.get() == stmt_id.get() &&
							range_stmt.get() == offset_base_stmt.get())
							is_positive_offset = true;
						else if (range_stmt.get() == stmt_id.get() &&
								 domain_stmt.get() == offset_base_stmt.get())
							is_positive_offset = false;
						else
							return isl::stat::ok();

						lrs_relation_found = true;

						ISL_DEBUG(std::cerr << "map "; isl_map_dump(map.get()));

						isl::union_map applied = map.to_union_map();
						applied = applied.apply_domain(tagged_partial_schedule);
						applied = applied.apply_range(tagged_partial_schedule);
						ISL_DEBUG(std::cerr << "applied ";
								  isl_union_map_dump(applied.get()));
						isl::map applied_map = applied.as_map();

						auto offset_set =
							isl::manage(isl_map_deltas(applied_map.copy()));
						ISL_DEBUG(std::cerr << "offset_set ";
								  isl_set_dump(offset_set.get()));

						isl::val offset_val =
							offset_set.plain_get_val_if_fixed(isl::dim::set, 0);
						isl_assert(ctx,
							offset_val.is_int() && "must be an integer",
							abort());
						int num = offset_val.get_num_si();
						int den = offset_val.get_den_si();
						int cur_offset = num / den;
						ISL_DEBUG(
							std::cerr << "offset " << cur_offset << std::endl);

						if (domain_stmt.get() == range_stmt.get())
							isl_assert(ctx,
								cur_offset == 0 &&
									"offset must be 0 for intra relations",
								abort());
						if (offset != invalid_offset)
							isl_assert(ctx,
								cur_offset == offset && "conflicting offsets",
								abort());

						offset = cur_offset;

						if (!is_positive_offset)
							offset = -offset;

						return isl::stat::ok();
					});

					if (is_intra_relation)
						isl_assert(ctx,
							!lrs_relation_found &&
								"Currently we do not support intra relations "
								"in array expanded bands",
							abort());

					if (!lrs_relation_found)
						offset = 0;

					isl_assert(ctx,
						offset != invalid_offset && "no valid offset found",
						abort());

					return offset;
				}();

				isl::aff aff;
				if (factor == 1) {
					aff = isl::aff::zero_on_domain(map.domain().space());
					isl_assert(ctx, offset == 0, abort());
				} else {
					aff = isl::aff::var_on_domain(
						isl::manage(isl_local_space_from_space(
							map.domain().space().release())),
						isl::dim::set, member);
				}

				if (offset != 0)
					aff = aff.add_constant_si(offset);
				if (factor != 1)
					aff = aff.mod(factor);
				ISL_DEBUG(std::cerr << "aff: "; isl_aff_dump(aff.get()));

				auto as_map = aff.as_map();
				as_map = as_map.add_dims(
					isl::dim::out, n_expanded_dims - 1 - depth - member);
				as_map = isl::manage(isl_map_insert_dims(
					as_map.release(), isl_dim_out, 0, depth + member));

				as_map = as_map.set_range_tuple(array_id);

				isl::space space = as_map.get_space();
				space = space.drop_dims(isl::dim::in, 0, n_members);
				space = space.range().map_from_domain_and_range(space.domain());
				space = space.add_dims(isl::dim::out, n_stmt_dims);
				space = space.set_tuple_id(isl::dim::out, stmt_id);
				auto tag = isl::multi_aff::domain_map(space);
				as_map = as_map.preimage_range(tag);

				auto as_umap = as_map.to_union_map();
				ISL_DEBUG(std::cerr << "as_umap: ";
						  isl_union_map_dump(as_umap.get()));
				if (umap.is_null())
					umap = as_umap;
				else
					umap = umap.unite(as_umap);

				return isl::stat::ok();
			});

			if (member_array_umap.is_null())
				member_array_umap = umap;
			else
				member_array_umap = member_array_umap.unite(umap);

			return isl::stat::ok();
		};
		if (isl_id_to_id_foreach(array_expansion, lambda1).is_error())
			return nullptr;

		if (full_array_umap.is_null())
			full_array_umap = member_array_umap;
		else
			// full_array_umap = full_array_umap.intersect(member_array_umap);
			full_array_umap = full_array_umap.intersect(member_array_umap);
	}

	ISL_DEBUG(std::cerr << "full_array_umap: ";
			  isl_union_map_dump(full_array_umap.get()));

	extra_umap = extra_umap.unite(full_array_umap);

	ISL_DEBUG(std::cerr << "computed extra_umap: ";
			  isl_union_map_dump(extra_umap.get()));

	return extra_umap.release();
}

extern "C" isl_union_set *isl_union_set_to_tags(isl_union_set *uset_)
{
	isl::union_set uset = isl::manage(uset_);
	isl::union_set tag_uset =
		isl::manage(isl_union_set_empty(uset.space().release()));
	uset.foreach_set([&](isl::set set) {
		isl::space space = set.space();
		isl::set tag_set =
			space
				.drop_dims(isl::dim::set, 0,
					unsignedFromIslSize(space.dim(isl::dim::set)))
				.set_tuple_id(isl::dim::set, space.tuple_id(isl::dim::set))
				.universe_set();
		tag_uset = tag_uset.unite(tag_set.to_union_set());
		return isl::stat::ok();
	});
	return tag_uset.release();
}
