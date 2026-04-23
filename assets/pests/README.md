# Pest Asset Library

Downloaded 3D pest assets live here. Large model and texture files are ignored
by git, while this README and `CREDITS.md` document the expected structure.

Recommended local layout:

```text
assets/pests/
  rodent/
    scary_ratmouse/
      model.glb
      textures/
  cockroach/
    ck_cockroach/
      model.glb
      textures/
  CREDITS.md
```

Use `.glb` or `.gltf` when Sketchfab offers them because texture references are
usually more reliable. `.fbx` is also acceptable. The renderer should fall back
to procedural pests if these local files are missing.
