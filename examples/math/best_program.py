prompt = """
Solve from first principles with explicit checks. Requirements:

1) Model precisely:
- Define all objects, variables, and constraints algebraically/combinatorially.
- Choose one counting model (labeled vs indistinguishable) and stay consistent. For combinatorics, either label and divide at the end OR keep indistinguishable throughout—do not mix.
- For number-theory/decimal/ratio problems, state factorizations and gcd/lcm relations explicitly.

2) Mapping/Counting rigor:
- When mapping elements between sets (e.g., m ↦ m/gcd(m,N)), prove injectivity/surjectivity or otherwise handle overlaps via inclusion–exclusion. Do not assume unions over divisors are disjoint without proof.
- When computing a probability, ensure numerator and denominator are counts from the same sample space.
- Keep all computations exact (fractions/radicals/modular arithmetic); avoid decimals unless terminating.

3) Geometry workflow:
- Draw and name a diagram (mentally or on paper). List candidate theorems: power of a point, radical axis, homothety, parallel chords/tangents, similar triangles, right triangles, cyclicity, angle/length chasing.
- Identify perpendiculars to tangents through centers; use rectangles formed by distances from a point on a circle to parallel lines; note that for a point on an incircle, distances to each of a pair of parallel sides sum to the diameter; use midpoint/radical-axis facts for intersecting circles and common tangents.
- Prefer exact relations (e.g., MP·MQ = (tangent length)^2) over coordinate guesses. If coordinates are used, ensure constraints (parallelism, perpendicularity, tangency) are enforced exactly.

4) Sanity checks and diagnostics:
- If an assumption yields a contradiction (e.g., negative squared length), discard and rebuild the setup.
- For combinatorics/NT counts, validate with a smaller analog (e.g., replace 9999 by 9 or 99) to detect double-counting/missing cases before scaling up.
- For expressions of the form m√n, reduce n to be squarefree; then compute the requested function (e.g., m+n).
- Perform at least one independent cross-check (alternative derivation, structural identity, modular check, or small-n analog).

5) Output:
- Extract exactly what is asked (e.g., remainder, perimeter, m+n). Provide the final answer only as a single number with no extra text.
"""
