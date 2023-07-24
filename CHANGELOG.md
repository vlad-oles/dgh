# Changelog

<!--next-version-placeholder-->
## v0.1.0 (Jul 24 2023)

### Feature

- Added searching for best c as default behavior
- Added function for random generation of a bi-mapping polytope vertex R
- Added `__version__` variable
- Python >= 3.8 is now required

### Fix

- Removed convexity-based calculation of c and warm-start sequences of c due to inefficiency

### Documentation

- Created "Advanced" section where the logic behind parameter c is explained

## v0.0.8 (May 11 2023)

### Feature

- Added f, g to verbose output if returning them

### Fix

- Corrected (halved) ub in summary (verbose > 0)

### Documentation

- Corrected project homepage
- Switched to url for the example illustration 

## v0.0.7 (May 10 2023)

### Feature

- Simplified import structure
- Added "summary" verbose level

### Documentation

- Updated example illustration

## v0.0.6 (May 9 2023)

### Feature

- Enabled setting lower bound to avoid redundant iterations
- Added flag for validating the triangle inequality

### Fix

- Fixed import structure
- Distances are scaled to prevent overflow

### Documentation

- Corrected library import in README.md

## v0.0.5 (May 7 2023)

### Fix

- Removed redundant `__init__.py`
- Renamed source directory

## v0.0.4 (May 7 2023)

### Fix

- Allowed for importing (oops)
- Corrected dimensions in `fg_to_R`

### Feature

- Different levels of verbosity

### Documentation

- Expanded README.md

## v0.0.3 (May 6 2023)

### Fix

- Prevented overflow from c**2

### Feature

- Switched to global iteration budget with no limitations per restart

### Documentation

- Added this changelog

## v0.0.2 (May 6 2023)

- First release of `dgh`