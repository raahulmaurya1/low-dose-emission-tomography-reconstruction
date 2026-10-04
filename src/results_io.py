# results_io.py: save/load result rows as CSV (csv module; resumable, atomic writes).
# Takes: a path under results/ and a list of row dicts.
# Returns: save_rows -> nothing (writes the CSV); load_rows -> list of row dicts.
