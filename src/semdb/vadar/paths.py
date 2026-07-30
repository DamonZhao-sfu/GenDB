#!/usr/bin/env python3
"""
vadar/paths.py — corpus path resolution for the operator library.

A corpus manifest's image column holds anything from an absolute path to a bare
filename, so every consumer that turns a row into an `ImagePatch` needs the same
resolution rule. It lives here, in `vadar/`, because generated programs call it
directly; the endpoint-backed tools outside the package import it from here too.
"""
import os


def resolve_image_path(uri, image_dir):
    """Resolve an image reference to a real file. Handles: an already-absolute path
    (animals), a path relative to a base — full repo-relative under the sembench root
    (cars/medical) or a bare filename under an images/ dir (mmqa)."""
    uri = str(uri)
    if os.path.isabs(uri) and os.path.exists(uri):
        return uri
    if image_dir:
        for cand in (os.path.join(image_dir, uri), os.path.join(image_dir, os.path.basename(uri))):
            if os.path.exists(cand):
                return cand
    return uri  # last resort: assume the column already holds a usable path
