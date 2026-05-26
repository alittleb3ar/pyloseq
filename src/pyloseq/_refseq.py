"""Reference sequence container (DNA sequences keyed by taxon ID).

R reference: phyloseq::refseq(object)
"""

from __future__ import annotations

from pathlib import Path

import skbio
import skbio.io  # noqa: F401 — ensures skbio.io is registered


class RefSeq:
    """Wraps a dictionary of reference sequences keyed by taxon ID.

    R reference: phyloseq::refseq(object)
    """

    def __init__(self, data: dict[str, skbio.DNA]) -> None:
        self._seqs: dict[str, skbio.DNA] = dict(data)

    @classmethod
    def from_fasta(cls, path: str | Path) -> RefSeq:
        """Load sequences from a FASTA file.

        R reference: readDNAStringSet() then RefSeq(x)
        """
        seqs: dict[str, skbio.DNA] = {}
        for seq in skbio.io.read(str(path), format="fasta", constructor=skbio.DNA):
            seqs[str(seq.metadata["id"])] = seq
        return cls(seqs)

    def to_fasta(self, path: str | Path) -> None:
        """Write sequences to a FASTA file.

        R reference: writeXStringSet(refseq(x), filepath)
        """
        with open(path, "w") as fh:
            for seq in self._seqs.values():
                skbio.io.write(seq, format="fasta", into=fh)

    @property
    def names(self) -> list[str]:
        """Taxon identifiers for all stored sequences.

        R reference: names(refseq(x))
        """
        return list(self._seqs.keys())

    def copy(self) -> RefSeq:
        """Return a deep copy of this RefSeq."""
        return RefSeq({k: skbio.DNA(str(v)) for k, v in self._seqs.items()})

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, key: str) -> skbio.DNA:
        return self._seqs[key]

    def __repr__(self) -> str:
        return f"RefSeq({len(self._seqs)} sequences)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RefSeq):
            return NotImplemented
        if set(self.names) != set(other.names):
            return False
        return all(str(self._seqs[n]) == str(other._seqs[n]) for n in self.names)
