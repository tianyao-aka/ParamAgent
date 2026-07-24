import csv
from typing import Union

class Game24Dataset:
    def __init__(self, file_path: str, delimiter: str = None):
        """
        Args:
            file_path: Path to your CSV/TSV file.
            delimiter: Field delimiter (e.g. ',' for CSV or '\\t' for TSV).
                       If None, will sniff from the file.
        """
        # Auto-detect delimiter if not provided
        if delimiter is None:
            with open(file_path, newline='') as f:
                sample = f.read(1024)
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
                delimiter = dialect.delimiter

        self.puzzles = []
        with open(file_path, newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # row['Puzzles'] is like "1 1 4 6"
                parts = row['Puzzles'].split()
                # produce "1,1,4,6"
                puzzle_str = ",".join(parts)
                self.puzzles.append(puzzle_str)

    def __len__(self) -> int:
        """Number of puzzles in the file."""
        return len(self.puzzles)

    def __getitem__(self, idx: Union[int, slice]) -> Union[str, list[str]]:
        """
        Get one puzzle (as '1,1,4,6') or a slice of puzzles.
        Raises IndexError for out-of-bounds ints.
        """
        return self.puzzles[idx]
