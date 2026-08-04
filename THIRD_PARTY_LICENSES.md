# Third-Party Licenses

This package is released under the MIT License (see `LICENSE`). It also
incorporates third-party components under the BSD 3-Clause License, reproduced
below as required by that license. Research-paper attributions for the
algorithms implemented here are in `doc/REFERENCES.md`.

---

## fast_gicp

`src/nano_gicp/nano_gicp.{h,cc}` (the `NanoGICP` registration engine) is a fork
of **fast_gicp** by SMRT-AIST (Kenji Koide et al.), extended in this repository
with a degeneracy gate and intensity/visual photometric terms.

- Upstream: https://github.com/koide3/fast_gicp (SMRT-AIST/fast_gicp)
- Reference: K. Koide, M. Yokozuka, S. Oishi, A. Banno, *Voxelized GICP for Fast
  and Accurate 3D Point Cloud Registration*, ICRA 2021.

```
BSD 3-Clause License

Copyright (c) 2020, SMRT-AIST
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## nanoflann

`include/nano_gicp/nanoflann.h`, `include/nano_gicp/nanoflann_adaptor.h`, and
`src/nano_gicp/nanoflann.cc` vendor **nanoflann** (a header-only kd-tree) with a
custom `SO3_Adaptor`. The full BSD license text is retained verbatim in the
header of each of those files; it is summarized here for discoverability.

- Upstream: https://github.com/jlblancoc/nanoflann
- Copyright (c) 2008–2009 Marius Muja; (c) 2008–2009 David G. Lowe;
  (c) 2011–2022 Jose Luis Blanco (Universidad de Malaga). BSD License.
- See the in-source file headers for the complete license text.
