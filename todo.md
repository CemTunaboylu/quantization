Todos:
- [ ] nonlinear quantizations
- [ ] 1-bit 1,6 bits

So:
- I only implemented uint8 packing. I did not test other dtypes, at this point it is a waste of time. I saw that uint16 to uint64 does not implement cpu right shift so it is a problem.