# WebArena site harness (local prove-out: Reddit only)

Stands up the WebArena **Reddit / forum (Postmill)** site in Docker so ModalityBench can drive
real long-horizon tasks against it. Reddit is small and light enough to run in Docker Desktop
on a laptop, so you can validate the whole live path — `WebArenaSource` → agent loop →
evict-vs-accumulate → scoring — at **$0** before committing to a paid VM for the heavier sites.

Everything here is written so the same files drop onto a Linux VM unchanged; the heavier sites
are left out of `docker-compose.yml` on purpose (see *Scaling up* below).

## Prerequisites
- Docker Desktop (Compose v2). `docker compose version` should work.
- The `[browser]` extra + Chromium for the auth capture and the run:
  `pip install -e '.[browser]' && playwright install chromium`
- An API key for the model + judge (e.g. `ANTHROPIC_API_KEY`).

## One-time setup
Run from this directory (`docker/webarena/`); scripts `cd` to their own location.

```bash
./scripts/download-images.sh     # ~a few GB from the CMU mirror (resumable)
./scripts/load-images.sh         # docker load the tar
./scripts/fetch-config.sh        # WebArena's test.raw.json (all 812 tasks; we filter to reddit)
```

## Each session
```bash
./scripts/up.sh                              # start forum + wait until it serves
# from the repo root:
source docker/webarena/scripts/print-env.sh     # exports REDDIT=http://localhost:9999
python docker/webarena/scripts/capture-auth.py  # REQUIRED: all 129 reddit tasks are login-gated
python -m modalitybench.cli run configs/webarena-reddit.yaml
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
