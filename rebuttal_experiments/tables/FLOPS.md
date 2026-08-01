# Approximate FLOPs (analytic, MLP 3x128, simulation-free CFM)

FM methods (Gaussian FM, source-only, RAFM) share the same MLP -> identical per-forward FLOPs; RAFM slerp+tangent projection add O(d)/sample (negligible). Forward ~= 2*params/sample. Training (1 fwd+1 bwd/step) ~= 6*params*batch*steps (steps=10000). Sampling ~= 2*params*NFE*Ngen (NFE=512 = RK4x128, Ngen=10000).

MSGM: O(d^2-d^3)/step (dxdxd drift tensor + Hutchinson-trace SSM); analytic FLOPs not derived, wall-clock proxy = ~540x RAFM training time.

| config | params | train FLOPs | sample FLOPs |
|---|---|---|---|
| Student-t d16 | 37,392 | 9.2TFLOP | 382.9GFLOP |
| Student-t d32 | 41,504 | 10.2TFLOP | 425.0GFLOP |
| PIV d64 | 49,728 | 12.2TFLOP | 509.2GFLOP |
| Finance d49 | 45,873 | 11.3TFLOP | 469.7GFLOP |
| Weather d96 | 57,952 | 7.1TFLOP | 593.4GFLOP |
| Student-t d256 | 99,072 | 24.3TFLOP | 1.0TFLOP |
