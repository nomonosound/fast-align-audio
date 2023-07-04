* Bump version in `fast_align_audio/__init__.py`
* Update CHANGELOG.md
* Commit and push the change with a commit message like this: "Release vx.y.z" (replace x.y.z with the package version)
* Wait for build workflow in Github Actions to complete
* Download wheels artifact from the build workflow
* Place all the fresh whl files in dist/
* `python -m twine upload dist/*`
