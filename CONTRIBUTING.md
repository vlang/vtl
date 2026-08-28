## Example Workflow for Contributing

(provided by [@spytheman](https://github.com/spytheman))

(If you don't already have a GitHub account, please create one. Your GitHub
username will be referred to later as 'YOUR_GITHUB_USERNAME'. Change it
accordingly in the steps below.)

1. Fork https://github.com/vlang/vtl using GitHub's interface to your own account.
   Let's say that the forked repository is at
   `https://github.com/YOUR_GITHUB_USERNAME/v` .
2. Clone the main vtl repository https://github.com/vlang/vtl to a local folder on
   your computer, say named vtl/ (`git clone https://github.com/vlang/vtl vtl`)
3. `cd vtl`
4. `git remote add pullrequest https://github.com/YOUR_GITHUB_USERNAME/v`
   NB: the remote named `pullrequest` should point to YOUR own forked repo, not the
   main v repository! After this, your local cloned repository is prepared for
   making pullrequests, and you can just do normal git operations such as:
   `git pull` `git status` and so on.

5. When finished with a feature/bugfix/change, you can:
   `git checkout -b fix_alabala`
6. `git push pullrequest` # (NOTE: the `pullrequest` remote was setup on step 4)
7. On GitHub's web interface, go to: https://github.com/vlang/vtl/pulls

   Here the UI shows a dialog with a button to make a new pull request based on
   the new pushed branch.
   (Example dialog: https://url4e.com/gyazo/images/364edc04.png)

8. After making your pullrequest (aka, PR), you can continue to work on the
   branch `fix_alabala` ... just do again `git push pullrequest` when you have more
   commits.

9. If there are merge conflicts, or a branch lags too much behind VTL's master,
   you can do the following:

   1. `git pull --rebase origin master` # solve conflicts and do
      `git rebase --continue`
   2. `git push pullrequest -f` # this will overwrite your current remote branch
      with the updated version of your changes.

The point of doing the above steps, is to never directly push to the main VTL
repository, _only to your own fork_. Since your local `master` branch tracks the
main VTL repository's master, then `git checkout master`, as well as
`git pull --rebase origin master` will continue to work as expected
(these are actually used by `v up`) and git can always do it cleanly.

Git is very flexible, so there are other ways to accomplish the same thing.
See the
[GitHub flow](https://guides.github.com/introduction/git-handbook/#github)
, for more information.

## Using Github's hub CLI tool

You can download the `hub` tool from https://hub.github.com/ . Using
`hub`, you will not need to go through the (sometimes) slow website
to make PRs. Most remote operations can be done through the `hub` CLI
command.

NB: You still need to have a GitHub account.

### Preparation:

(steps 1..3 need to be done just _once_):

1. `hub clone vlang/vtl my_vtl`
2. `cd my_vtl`
3. `hub fork --remote-name pullrequest`

4. `git checkout -b my_cool_feature` # Step 4 is better done _once per each new
   feature/bugfix_ that you make.

### Improve VTL by making commits:

5. `git commit -am "math: add a new function copysign"`

### Testing your commits locally:

You can test locally whether your changes have not broken something by
running: `./bin/test`. See `README.md` for more details.

### Publishing your commits to GitHub:

6. `git push pullrequest`

### Making a PR with `hub`:

(so that your changes can be merged to the main VTL repository)

7. `hub pull-request`

Optionally, you can track the status of your PR CI tests with:

8. `hub ci-status --verbose`

### Fixing failing tests:

If everything is OK, after some minutes, the CI tests should pass for
all platforms. If not, visit the URLs for the failing CI jobs, see
which tests have failed and then fix them by making more changes. Just use
`git push pullrequest` to publish your changes. The CI tests will
run with your updated code. Use `hub ci-status --verbose` to monitor
their status.

## Coding standards

- Run `v fmt -w .` before committing; CI runs `v fmt` checks.
- Prefer clear, descriptive names over clever abbreviations. Public API
  functions are `snake_case` like the rest of V.
- Keep functions small and focused; document public API with `fn` doc
  comments where the intent is not obvious from the name.
- Follow the surrounding file's style — VTL mixes tensor ops, math, and
  autograd, so match the conventions of the module you touch.

## Test-writing conventions

- Tests live in `tests/` and use the `_test.v` suffix (e.g.
  `tests/math_op_test.v`). A test file for `foo.v` is `foo_test.v` next to
  it or in `tests/` for integration-style coverage.
- Use `assert` with a clear message; for float comparisons use a tolerance
  (`assert abs(a - b) < 1e-6`) instead of exact equality — tensor math is
  not bit-exact across backends.
- When a test is expected to fail or is backend-specific (CUDA/Vulkan),
  gate it with `$if cuda ? { ... }` blocks or a skip pattern rather than
  deleting it. Broken CI on one backend should be visible, not silent.
- Run the full suite locally before pushing: `./bin/test` (see README).

## Commit message format

- Prefix commits by area: `math:`, `nn:`, `autograd:`, `la:`, `docs:`,
  `tests:`, `ci:` — e.g. `math: add a new function copysign`.
- One logical change per commit; reference the issue number when one
  exists (`nn: fix dropout mask for eval mode (#123)`).
- Keep the message imperative and under ~72 characters for the subject.

## Adding a new layer / optimizer / loss

1. Place the implementation in the matching module (`nn/`, `autograd/`,
   `ml/`), following the pattern of the closest existing layer.
2. Add a `_test.v` covering forward, backward (if autograd), shape
   mismatches, and a tolerance-based numerical check.
3. Wire it into the module's public re-exports if other layers are exposed
   there.
4. Update `README.md` / docs when the layer is part of the public API.

## Required CI gates before PR

All of these must pass for a PR to be merge-ready:

- `./bin/test` locally (or the equivalent `v test` on the modules you changed)
- `v fmt` (formatting check)
- The `.github/workflows/ci.yml` job matrix (multiple platforms/backends)
- `ci-full-ml.yml` for changes touching `ml/` or `nn/`

Use `hub ci-status --verbose` after pushing to track the jobs; fix failures
and push again rather than dismissing a red job.

## Hacktoberfest

Hacktoberfest-related PRs follow the same process and gates above. A PR
with a valid `hacktoberfest-accepted` label is one that has passed CI and
was reviewed — quality rules are identical regardless of the month.
