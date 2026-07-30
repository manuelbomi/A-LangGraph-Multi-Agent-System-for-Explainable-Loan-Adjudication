# Running on OpenShift

The manifests in `deploy/k8s/` are plain upstream Kubernetes and work on
OpenShift largely as-is, with a handful of platform differences worth
calling out explicitly for a reviewer evaluating this repo against an
OpenShift-based internal platform (common at large regulated enterprises).

## Security Context Constraints (SCC) vs. Pod SecurityContext

OpenShift enforces **SCCs** in addition to (and often instead of relying
solely on) the Pod-level `securityContext` fields used in
`deploy/k8s/deployment.yaml`. Specifically:

- The default `restricted` (or `restricted-v2`) SCC assigns a **random
  UID** to the container at runtime rather than honoring a fixed
  `runAsUser`. This repo's `Dockerfile` creates a dedicated `app` user, but
  does **not** hardcode `runAsUser` in the Deployment for exactly this
  reason -- forcing a specific UID would conflict with OpenShift's SCC
  admission and fail to schedule under the default policy.
- Because the container may run under an arbitrary assigned UID, the
  `/app/data` directory (where the SQLite checkpoint file lives) must be
  group-writable, not just owner-writable. The Dockerfile's `chown -R
  app:app /app` should be supplemented with `chmod g+w /app/data` (or an
  `initContainer`/`fsGroup` on the PVC) before deploying under a
  non-default SCC on OpenShift.
- `capabilities: drop: ["ALL"]` and `allowPrivilegeEscalation: false` (both
  already set in `deployment.yaml`) align with OpenShift's `restricted`
  SCC out of the box and should not need adjustment.

## DeploymentConfig vs. Deployment

OpenShift historically used its own `DeploymentConfig` resource (with
built-in triggers for image-stream changes and richer rollout hooks)
alongside standard Kubernetes `Deployment` objects. Modern OpenShift
(4.x+) fully supports plain `Deployment` resources, which is what this repo
ships -- there is no need to port `deploy/k8s/deployment.yaml` to a
`DeploymentConfig` for current OpenShift versions. If integrating with an
older OpenShift 3.x-style pipeline that expects `DeploymentConfig` and
`ImageStream` triggers (auto-redeploy on a new image push to the internal
registry), that would be an additive manifest, not a replacement.

## Routes vs. Ingress

`deploy/k8s/service.yaml` is a `ClusterIP` Service. OpenShift's `Route`
resource (roughly analogous to, but distinct from, a Kubernetes `Ingress`)
would be layered on top for external access, typically with edge TLS
termination handled by the platform's router rather than in-app -- this
repo's FastAPI app does not terminate TLS itself in any deployment target.

## Image builds

This repo's `Dockerfile` builds with any standard OCI-compatible builder.
On OpenShift, that image can be built via `oc new-build --binary` /
`BuildConfig` + `ImageStream` (OpenShift's native S2I-style build path) as
an alternative to building externally (e.g. in the GitHub Actions CI
pipeline at `.github/workflows/ci.yml`) and pushing to an image registry
OpenShift then pulls from. Both are valid; this repo's CI assumes the
latter (external build + registry push) as the more portable default.
