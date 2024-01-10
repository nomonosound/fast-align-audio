* Bump version in `fast_align_audio/__init__.py`
* `python setup.py develop && pytest`
* Update CHANGELOG.md
* Commit and push the change with a commit message like this: "Release vx.y.z" (replace x.y.z with the package version)
* Wait for build workflow in GitHub Actions to complete
* Download & extract wheels artifacts from the build workflow
* Place all the fresh whl files in dist/
* `python -m twine upload dist/*`
* Add a tag with name "vx.y.z" to the commit
* Go to https://github.com/nomonosound/fast-align-audio/releases and create a release where you choose the new tag
