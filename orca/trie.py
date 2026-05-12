"""
Trie index for reverse lookup: call sequence → call type decomposition.

Given a sequence label, the trie finds all possible call-type prefixes,
enabling decomposition into constituent calls.
"""


class TrieNode:
    __slots__ = ("children", "root_ids")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.root_ids: list[int] = []


class Trie:
    """A trie of all call-type strings for prefix matching."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, root_id: int):
        """Insert a call type into the trie."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.root_ids.append(root_id)

    def find_prefixes(self, word: str) -> list[tuple[int, str]]:
        """
        Find all call types that are prefixes of the given string.

        Returns list of (root_id, remainder) tuples.
        """
        results = []
        node = self.root
        for i, ch in enumerate(word):
            if ch not in node.children:
                break
            node = node.children[ch]
            if node.root_ids:
                remainder = word[i + 1:]
                for rid in node.root_ids:
                    results.append((rid, remainder))
        return results


def build_trie(kernel) -> Trie:
    """Build a trie from all call types in the kernel."""
    trie = Trie()
    for root in kernel.all_roots():
        trie.insert(root.call_type, root.id)
    return trie
