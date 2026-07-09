# WebArena site harness (Reddit only)

Stands up the WebArena **Reddit / forum (Postmill)** site in Docker so ModalityBench can drive
real long-horizon tasks against it. Reddit is the lightest WebArena site (~129 tasks), so it's
the cheapest way to validate the whole live path — `WebArenaSource` → agent loop →
evict-vs-accumulate → scoring — before standing up a VM for the heavier sites.

Everything here is written so the same files drop onto a Linux VM unchanged; the heavier sites
are left out of `docker-compose.yml` on purpose (see *Scaling up* below).

> ⚠️ **Requires an amd64 / x86-64 host.** The WebArena images are published **amd64-only**. On an
> ARM64 machine (Apple Silicon, Windows-on-ARM) Docker runs them under QEMU emulation, where the
> Postmill PHP workers crash with `QEMU internal SIGILL` — the site never stays up. Use an
> amd64 desktop or an amd64 cloud VM. The image is ~50 GB ("withimg" = populated with post
> images), so the one-time `docker load` takes a while and needs ~60 GB of free disk.

## Prerequisites
- An **amd64 host** (see the warning above) with Docker Desktop / Engine (Compose v2).
  `docker compose version` should work.
- The `[browser]` extra + Chromium for the auth capture and the run:
  `pip install -e '.[browser]' && playwright install chromium`
- An API key for the model + judge (e.g. `ANTHROPIC_API_KEY`).

> **Windows:** use the `.ps1` scripts from **PowerShell** (not cmd.exe). Every `.sh` below has a
> `.ps1` twin — e.g. `.\scripts\download-images.ps1`. On macOS/Linux use the `.sh` versions from
> bash. `capture-auth.py` is plain Python and works on both.

## One-time setup
Run from this directory (`docker/webarena/`); scripts `cd` to their own location.

```bash
# bash (macOS/Linux/Git Bash)            # PowerShell (Windows)
./scripts/download-images.sh             # .\scripts\download-images.ps1   (~few GB, resumable)
./scripts/load-images.sh                 # .\scripts\load-images.ps1
./scripts/fetch-config.sh                # .\scripts\fetch-config.ps1
```

## Each session
```bash
# bash                                   # PowerShell
./scripts/up.sh                          # .\scripts\up.ps1
# from the repo root:
source docker/webarena/scripts/print-env.sh          # . docker\webarena\scripts\print-env.ps1
python docker/webarena/scripts/capture-auth.py       # (same) REQUIRED — reddit tasks need login
python -m modalitybench.cli run configs/webarena-reddit.yaml   # (same)
```

All 129 reddit tasks are login-gated and most (117) are scored by `program_html`, which
re-fetches in a fresh **authenticated** context — so the `capture-auth.py` step is required, not
optional. `./scripts/reset.sh` restores pristine site state between runs (recreates the container from the
pre-populated image — needed for tasks that post/vote/edit). `docker compose stop` to pause;
`docker compose down` to remove the container (image stays loaded, so restart is fast).

## What each piece does
| File | Role |
|---|---|
| `docker-compose.yml` | the `forum` service (port 9999→80), `pull_policy: never` |
| `scripts/download-images.sh` | fetch the Postmill tar (CMU mirror; `FORUM_URL` overrides) |
| `scripts/load-images.sh` | `docker load` the tar |
| `scripts/fetch-config.sh` | download WebArena's raw task configs |
| `scripts/up.sh` / `wait-ready.sh` | start + health-poll |
| `scripts/reset.sh` | recreate container = pristine seeded site |
| `scripts/print-env.sh` | `source` it to export `REDDIT` for ModalityBench |
| `scripts/capture-auth.py` | Playwright login → `.auth/reddit_state.json` |

`downloads/`, `.auth/`, `config_files/`, and `.env` are gitignored (huge / secret / generated).

## Notes
- Credentials are WebArena's public test account (`MarvelsGrantMan136` / `test1234`), overridable
  via `REDDIT_USER` / `REDDIT_PASS`. Not secret, but the captured cookie is — hence `.auth/` is
  ignored.
- Postmill uses relative URLs, so `WEBARENA_HOST=localhost` is fine locally. On a VM, set it to a
  hostname the browser can reach if you hit absolute-URL redirects.

## Scaling up (later, on a VM)
To add heavier sites, uncomment their services in `docker-compose.yml` and add their tars +
post-start URL surgery (Magento `setup:store-config:set` base-url, GitLab `external_url`). The
`shopping`/`gitlab` images are tens of GB and RAM-hungry; budget a `t3a.xlarge`-class VM with a
~250 GB disk (skip `map` — it needs the ~1 TB dedicated setup). Cost is dominated by the disk
(~$20/mo for the volume even when stopped), not the compute.
