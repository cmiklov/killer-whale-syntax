"""
Interactive REPL for the orca-engine.

Usage:
  python -m orca              # interactive REPL
  python -m orca "S1"         # single query
  python -m orca "align:J:K"  # cross-pod alignment
"""

import sys
import json
from .navigate import OrcaEngine
from .kernel import OrcaKernel


BANNER = """
╔══════════════════════════════════════════════════════╗
║  orca — vocalisation topology navigator              ║
║  R/D analysis of orca communication                  ║
╚══════════════════════════════════════════════════════╝

Commands:
  S1              look up call type
  S1:S7           compound (S1 modifies S7)
  ?S1             reverse lookup (all pods)
  @3              look up by ID
  pod:J           all J-pod call types
  think:S1,S7     co-activate and relax
  align:J:K       cross-pod alignment
  field           field diagnostics
  S1.*            all compounds with S1 as modifier
  stats           lexicon statistics
  help            this message
  exit            quit
"""


def _format_result(result: dict) -> str:
    """Format a result dict for display."""
    lines = []

    if "error" in result:
        return f"  ERROR: {result['error']}"

    if "call_type" in result:
        r = result
        lines.append(f"  {r['call_type']}  (id={r['id']}, pod={r['pod']}, addr={r['address']})")
        if r.get("description"):
            lines.append(f"    {r['description']}")
        if r.get("contexts"):
            lines.append(f"    contexts: {', '.join(r['contexts'])}")
        if r.get("frequency"):
            lines.append(f"    frequency: {r['frequency']:.2f}")
        if r.get("nearest"):
            lines.append(f"    nearest: {', '.join(f'{ct}({sim})' for ct, sim in r['nearest'])}")

    elif "compound" in result and "nearest" in result:
        lines.append(f"  {result['compound']}")
        if isinstance(result["nearest"], list):
            lines.append(f"    → {', '.join(f'{ct}({sim})' for ct, sim in result['nearest'])}")
        else:
            lines.append(f"    → {result['nearest']} ({result.get('similarity', '')})")

    elif "pod" in result and "call_types" in result:
        lines.append(f"  Pod {result['pod']}: {result['count']} call types")
        for ct in result["call_types"]:
            ctx = ", ".join(ct.get("contexts", []))
            lines.append(f"    {ct['call_type']} (id={ct['id']}) [{ctx}]")

    elif "think" in result:
        lines.append(f"  think({', '.join(result['think'])})")
        lines.append(f"    emerged: {', '.join(f'{ct}({sim})' for ct, sim in result['emerged'])}")

    elif "alignment" in result:
        lines.append(f"  {result['alignment']}")
        lines.append(f"    disparity: {result['disparity']}")
        lines.append(f"    context agreement: {result['context_agreement']}")
        lines.append(f"    matched pairs: {result['matched_pairs']}")
        for corr in result.get("correspondences", []):
            lines.append(f"    {corr['source']} ↔ {corr['target']} ({corr['similarity']})")

    elif "phonosemantic_correlation" in result:
        lines.append(f"  Phonosemantic correlation: {result['phonosemantic_correlation']}")
        lines.append(f"    {result['interpretation']}")
        lines.append(f"    {result['n_roots']} call types, {result['dim']}D field")
        lines.append(f"    pods: {', '.join(result['pods'])}")

    elif "roots" in result:
        lines.append(f"  {result.get('compound_label', '')}")
        if result.get("pod"):
            lines.append(f"    pod: {result['pod']}")
        if result.get("description"):
            lines.append(f"    {result['description']}")

    else:
        lines.append(f"  {json.dumps(result, indent=2, default=str)}")

    return "\n".join(lines)


def _print_stats(engine: OrcaEngine):
    """Print lexicon statistics."""
    k = engine.kernel
    n = len(k)
    pods = k.pods()
    print(f"\n  Kernel: {n} call types across {len(pods)} pods")
    for pod in pods:
        count = len(k.by_pod(pod))
        print(f"    {pod}-pod: {count} discrete calls")
    print(f"  Two-call compounds: {n * (n-1):,}")
    print(f"  Three-call compounds: {n * (n-1) * (n-2):,}")
    corr = engine.field.phonosemantic_correlation()
    print(f"  Phonosemantic correlation: {corr:.4f}")
    print()


def repl(engine: OrcaEngine):
    """Interactive REPL."""
    print(BANNER)
    _print_stats(engine)

    while True:
        try:
            expr = input("orca> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  gre")
            break

        if not expr:
            continue
        if expr in ("exit", "quit", "gre"):
            print("  gre")
            break
        if expr == "help":
            print(BANNER)
            continue
        if expr == "stats":
            _print_stats(engine)
            continue

        results = engine.query(expr)
        for r in results:
            print(_format_result(r))
        print()


def main():
    """Entry point."""
    # Build engine with mock data for now
    from tests.conftest import _build_mock_roots
    kernel = OrcaKernel(roots=_build_mock_roots())
    engine = OrcaEngine(kernel=kernel)

    if len(sys.argv) > 1:
        expr = " ".join(sys.argv[1:])
        results = engine.query(expr)
        for r in results:
            print(_format_result(r))
    else:
        repl(engine)
