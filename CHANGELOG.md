# Changelog

## [Unreleased]

## 0.5.0 - (unreleased)

### Added

* Added `Regioncam1D`, `Plane1D`
* Added python interface for `Regioncam1D`
* Added `slice` method that extracts a 1d slice from a 2d regioncam
* Added `Regioncam.show` method to python interface that returns a rendered png image,
  this allows regioncam objects to be garbage collected in jupyter notebooks.
* Added `edge_length` and `face_area` (exposed as Edge.length and Face.area in python)

### Changed

* Generalized `Plane` to `Hyperplane`
* Renamed `Plane::inverse` to `Hyperplane::project`


## 0.4.1 - 2025-04-17

### Added

* Added `color` argument to `mark_points`
* Add text_only_markers variant
* Add option to change size of points through plane

### Changed

* `Regioncam::from_plane` now takes `size` argument


## 0.4.0 - 2025-04-13

### Added

* Piet based rendering backend, with direct png rendering (`write_png`).
* Added more options to customize rendering.
* Added option to render vertices

### Changed

* Face colors are based on hash of activation pattern, this makes colors stable even when other faces change.
* Remove `_repr_svg_` in python interface in favor of `_repr_png_`, this should result in smaller notebook files.


## 0.3.0 - 2025-02-12

### Added

* Construct a Plane from a Linear transformation
* Expose sequence of vertices, faces, edges, layers in python interface


## 0.2.0 - 2025-02-10

### Changed

* Split regioncam into partition (the halfedge data structure) and regioncam (which stores layer outputs)
* Don't print 0 size points/labels


## 0.1.1 - 2025-01-28

### Changed

* Reorganize code: make main crate the root crate of the workspace.