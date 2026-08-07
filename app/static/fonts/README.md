# Bundled fonts

Both families are licensed under the SIL Open Font License 1.1, which permits
redistribution alongside the app. They are vendored (rather than linked from a CDN)
so the page renders identically with no network access — the same promise the rest
of the app makes.

| Family | Files | Used for | Source |
| --- | --- | --- | --- |
| IBM Plex Sans Arabic | `ibm-plex-sans-arabic-{400,500,600,700}-{arabic,latin}.woff2` | interface text | <https://github.com/IBM/plex> |
| Noto Naskh Arabic | `noto-naskh-arabic-{400,600}-arabic.woff2` | the transcript pane | <https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic> |

The `.woff2` files are the Google Fonts subsets (Arabic and Latin ranges only); the
matching `unicode-range` declarations live in `../styles.css`.
