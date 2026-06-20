# Journal de bord — COLMAP dense sans CUDA (Snapdragon X Plus / WoA ARM64)

## Contexte & objectif

COLMAP `patch_match_stereo` est CUDA-only par construction : les kernels PatchMatch MVS
(`src/colmap/mvs/patch_match_cuda.cu`) SONT l'algorithme, et tout le module (y compris le
controller host-side) n'est compilé que sous `CUDA_ENABLED`. Objectif : un backend de
fallback CPU (puis GPU portable Adreno) intégré proprement dans COLMAP, buildable
ARM64 Windows pour un Zenbook UX3407Q (Snapdragon X Plus).

Machine cible : Snapdragon X Plus, 10 cœurs Oryon, Adreno GPU (OpenCL 3.0 / Vulkan),
NPU Hexagon (écarté : QNN/SNPE = graphes NN statiques, PatchMatch ne se mappe pas).

## Référence de code

- Repo : https://github.com/colmap/colmap
- **Commit de base : `f72d933868e3377b8bb5a96bf04b06e84f0403b4`**
  (2026-06-08, "Add GlobalMapper.keep_max_num_tracks option...")
- Patch à appliquer : `colmap-cpu-patch-match-v1.patch` (git apply depuis la racine du repo)

## Trace de la dépendance CUDA (vérifiée sur source, 2026-06-10)

1. `cmake/FindDependencies.cmake:192` — définit `COLMAP_CUDA_ENABLED` si CUDA trouvé.
2. `src/colmap/mvs/CMakeLists.txt` — `patch_match.cc` + `patch_match_cuda.cu` compilés
   uniquement dans `colmap_mvs_cuda` sous `if(CUDA_ENABLED)`. Le `PatchMatchController`
   (pur host) est lui aussi enfermé dans la lib CUDA.
3. `src/colmap/exe/mvs.cc:234` — `RunPatchMatchStereoImpl` : `#if !defined(COLMAP_CUDA_ENABLED)`
   → `LOG(FATAL_THROW) "Dense stereo reconstruction requires CUDA..."`. Dupliqué dans
   `ui/dense_reconstruction_widget.cc:410` et `controllers/automatic_reconstruction.cc:417`.
4. `PatchMatch::Run()` (`patch_match.cc:133`) instancie `PatchMatchCuda` en dur. Zéro abstraction.
5. Dépendances hardware réelles des kernels : textures layered CUDA (bilinéaire HW pour les
   images sources, point pour ref/depth/poses), `curand` (sampling Monte-Carlo), mémoire
   partagée (cache de l'image ref par warp). `stereo_fusion`, meshing : déjà 100 % CPU.

Alternatives ré-évaluées : COLMAP-CL mort (v1.5 ~2021, binaires x64 only, pas de source) ;
OpenMVS = option pragmatique CPU mais hors-COLMAP et sans backend GPU non-CUDA.

## Plan

- **Phase 0 — Refactor backend-agnostic** : interface `PatchMatchBackend`, factory runtime
  (`--PatchMatchStereo.backend {auto,cuda,cpu}`), sortir le controller de la lib CUDA,
  supprimer les 3 gates fatals. Nouvelle cible CMake `colmap_mvs_patch_match` (controller +
  wrapper + factory) pour casser la circularité mvs ↔ mvs_cuda.
- **Phase 1 — Backend CPU** : port algorithmique fidèle des kernels en C++17 + OpenMP
  (parallèle sur colonnes dans le sweep, sur lignes dans l'init). PCG32 par pixel à la place
  de curand. Sampler bilinéaire/border maison à la place des textures. Tables de lookup pour
  les poids bilatéraux (exp précalculés : grille spatiale + 256 diffs de couleur quantifiée).
- **Phase 2 — Backend GPU Adreno** (plus tard) : OpenCL 3.0 d'abord (port quasi mécanique,
  image2d_array_t + CLK_FILTER_LINEAR ≈ layered textures), Vulkan compute en plan B.
  Piège connu : TDR Windows → chunker le sweep en plusieurs launches.
- **Phase 3 — Validation & packaging ARM64** : comparaison statistique vs depth maps CUDA
  de référence (ETH3D low-res), build vcpkg `arm64-windows`, CLI d'abord.

## Décisions de design (Phase 0+1)

- `PatchMatchBackend` : interface pure { Run, GetDepthMap, GetNormalMap, GetSelProbMap,
  GetConsistentImageIdxs }. `PatchMatchCuda` la implémente sans autre changement.
- Sémantique répliquée à l'identique depuis le CUDA, y compris :
  - le schéma de rotation 90° CCW ×4 par itération (les buffers tournent, le sweep est
    toujours top→bottom) ;
  - l'adressage *border* des textures (couleur 0 hors image, y compris dans le filtre
    bilatéral de l'image ref — ça compte pour la parité aux bords) ;
  - la convention bilinéaire CUDA (`tex2D(x+0.5)` linéaire ≡ bilinéaire aux centres de
    pixels) et le nearest `floor(x+0.5)` pour les depth maps sources ;
  - l'image ref quantifiée uint8 (lecture `q/255` comme `cudaReadModeNormalizedFloat`) ;
  - l'ordre RNG init depth → init normals par pixel.
- RNG : PCG32 par pixel, seedé par index linéaire (splitmix64). Résultats **statistiquement**
  équivalents au CUDA, pas bit-exact (curand Philox ≠ PCG) — la validation devra être métrique.
- Fenêtre NCC : paramètres runtime (pas le dispatch template ×40 du CUDA — inutile sur CPU,
  le coût est dominé par le sampling bilinéaire et les exp, traités par lookup tables).
- `backend=auto` : cuda si compilé ET ≥1 device, sinon cpu (avec warning de lenteur).
- `gpu_index` ignoré par le backend CPU ; le controller utilise alors un seul worker et
  laisse OpenMP saturer les cœurs à l'intérieur d'un problème.
- AutomaticReconstruction : ne skippe plus le dense sans CUDA, log un warning et continue en CPU.

## Avancement

### 2026-06-10 — Session 1
- [x] Trace complet de la dépendance CUDA sur source (commit f72d933), assumptions du
      screenshot confirmées. COLMAP-CL confirmé mort.
- [x] Plan global posé (phases 0→3), NPU écarté avec justification.
- [x] Phase 0 : interface backend + factory + option `PatchMatchStereo.backend` +
      restructuration CMake (`colmap_mvs_patch_match`) + suppression des gates
      (exe, UI, automatic_reconstruction) + enregistrement CLI de l'option.
- [x] Phase 1 : `patch_match_cpu.{h,cc}` — port complet : filtre bilatéral ref,
      init cost, sweep (messages forward/backward, sampling Monte-Carlo, propagation,
      perturbation, geom consistency, filtrage), rotations, extraction des résultats.
- [x] Validation syntaxique des nouveaux fichiers (g++ -fsyntax-only, deps partielles).
      Build complet non réalisé dans la sandbox (deps lourdes) — à faire côté machine de dev.
- [x] Patch généré : `colmap-cpu-patch-match-v1.patch`.

### TODO prochaine session
- [ ] Build complet x64 Linux ou Windows (vcpkg) avec `-DCUDA_ENABLED=OFF`, fix des
      erreurs de compil restantes éventuelles.
- [ ] Smoke test sur un petit dataset (ex: south-building réduit, max_image_size 800,
      window_step 2, num_iterations 3) : photometric d'abord, puis geom.
- [ ] Comparaison métrique vs sortie CUDA de référence (completeness / erreur médiane
      des depth maps, nb de points après fusion).
- [ ] Perf : profiler le sweep ; si bound par accès colonne (stride), tester la
      transposition des maps pour le sweep. SIMD NEON explicite si besoin.
- [ ] Build ARM64 Windows via vcpkg arm64-windows (CLI only, GUI_ENABLED=OFF d'abord).
- [ ] Phase 2 OpenCL : extraire l'orchestration commune, écrire les kernels .cl.

## Notes / pièges rencontrés

- `Mat<T>` CPU est slice-major (`slice*W*H + row*W + col`), comme `GpuMat` (pitch par
  (slice,row)). La rotation CCW : `out(W_in-1-col, row) = in(row, col)`, dims swappées.
- Les normales sont d'abord tournées en composantes ((x,y,z)→(y,−x,z)) PUIS spatialement.
- Le sweep CUDA ne lit/écrit l'état RNG que sur la ligne 0 de chaque colonne — répliqué.
- `global_workspace_` CUDA (messages forward + probs de sampling) devient simplement
  deux vecteurs locaux par colonne.
- `kMaxPatchMatchWindowRadius = 32` venait de la shared memory CUDA — conservé tel quel
  pour ne pas diverger des options validées.
- Le filtre ref inclut les pixels hors-bord avec couleur 0 ET poids non nul (comportement
  texture border CUDA) — ne pas "corriger" ça, c'est la sémantique de référence.
- Divergence assumée vs CUDA dans `InitSourceImages` : le CUDA memcpy les images sources
  de façon contiguë dans des layers de stride max_width → contenu scramblé si les images
  sources ont des tailles hétérogènes (bug latent upstream ; les depth maps, elles, sont
  copiées ligne à ligne). Le port CPU copie ligne à ligne avec le stride paddé pour les
  deux cas. Identique pour des images de même taille (cas usuel après undistort).
- La quantification CUDA `uint8(255 * (q/255))` est l'identité pour q ∈ [0,255] (vérifié
  numériquement) — le port stocke directement q, strictement équivalent.
- `Mat<T>` n'a pas de `SetSlice` (contrairement à `GpuMatView`) — helper `SetSlice3` local.
- Validation : `g++ -std=c++17 -fsyntax-only -fopenmp -Wall -Wextra` OK sur tous les
  fichiers nouveaux/modifiés (avec headers glog/eigen/boost/ceres partiels), et le patch
  s'applique + compile (syntaxe) sur un clone propre de f72d933. Build complet impossible
  en sandbox (OpenImageIO, Metis, CHOLMOD, Glew, PoseLib, faiss, onnxruntime manquants).
