"""Where each released file lives.

This package publishes the artifact as typed trees (code/, data/, results/, docs/,
adjudication/) rather than as one directory. The scripts address their inputs by bare
filename, so this module is what turns a name into a path. `subdir` is the same rule the
package was laid out with, copied verbatim from the build, so the layout and this lookup
cannot disagree.

Nothing here is specific to a machine or a checkout: paths are derived from this file's own
location, so a script resolves its inputs the same way whatever the working directory is.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # package root

ROOT_FILES = {"README.md", "requirements.txt", "environment.lock.yml", "RESULT_SHA256SUMS.txt"}
DOC_FILES = {"REPRODUCE.md", "ARTIFACT_MANIFEST.md", "probability_space.md", "LAYOUT.md",
             "WITHHELD_FROM_RELEASE.txt"}


def subdir(name):
    """Directory of a released file inside the package, from its bare filename.

    Total by construction: a name the release does not carry (a withheld sheet, a result a
    script is about to write) still gets the one place it would live, so reads and writes
    agree without the file having to exist first.
    """
    if name in ROOT_FILES:
        return ""
    if name in DOC_FILES:
        return "docs"
    if name.endswith(".py"):
        return "code"
    if name.startswith("twins_"):
        return "data/twins"
    if name.startswith("scores_"):
        return "data/scores"
    if name.startswith(("natural_real_", "pilot_labeled_")) or name == "real_sources_questions.jsonl":
        return "data/streams"
    return "results"


def artifact(name):
    """Absolute path of a released file, from its bare name."""
    base = os.path.basename(name)
    d = subdir(base)
    return os.path.join(ROOT, d, base) if d else os.path.join(ROOT, base)
