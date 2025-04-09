from table_types import Datatable


def remove_overlapping_tables(tables: list[Datatable]) -> list[Datatable]:
    """
    Remove overlapping tables by keeping the larger one when significant overlap is detected.

    Args:
        tables: List of Datatable objects

    Returns:
        Filtered list of Datatable objects with overlapping tables removed
    """
    if not tables or len(tables) <= 1:
        return tables

    # Sort tables by area in descending order (largest first)
    sorted_tables = sorted(tables, key=lambda t: t.rect.get_area(), reverse=True)

    # Tables to keep after filtering
    filtered_tables = []
    removed_indices = set()

    for i, table in enumerate(sorted_tables):
        if i in removed_indices:
            continue

        filtered_tables.append(table)

        # Compare with all smaller tables
        for j in range(i + 1, len(sorted_tables)):
            if j in removed_indices:
                continue

            smaller_table = sorted_tables[j]

            # Check overlap
            overlap_rect = table.rect.get_overlap_rect(smaller_table.rect)

            if overlap_rect:
                # Calculate overlap percentage relative to the smaller table
                overlap_area = overlap_rect.get_area()
                smaller_area = smaller_table.rect.get_area()
                overlap_percentage = overlap_area / smaller_area

                # If over 90% of the smaller table overlaps with the larger one, remove it
                if overlap_percentage > 0.9:
                    removed_indices.add(j)

    return filtered_tables
