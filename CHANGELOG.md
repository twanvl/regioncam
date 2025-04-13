# Changelog

## [Unreleased]

## 0.4.0 - 2025-04-13

### Added

* Piet based rendering backend, with direct png rendering (`write_png`).
* Added more options to customize rendering.

### Changed

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