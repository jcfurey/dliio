# Ouster intensity and reflectivity

DLIIO decodes both sensor measurements independently, preserves them through
deskewing and voxel averaging, and supplies separate derived channels to the
photometric and LiDAR-image residuals.

| Measurement | Original Ouster cloud | Native Ouster cloud | Internal field |
|---|---|---|---|
| Return signal | `intensity` (float) | `signal` (integer) | `intensity` |
| Calibrated reflectivity | `reflectivity` (uint16) | `reflectivity` (profile-dependent integer) | `reflectivity` |
| Ambient near infrared | `ambient` | `near_ir` | Selected into `lidar_intensity` when requested |

The original driver's floating-point intensity stores the signal measurement;
its datatype does not imply an 8-bit range. Reflectivity is already compensated
by the sensor for range and sensitivity and must not receive a second raw-signal
range correction. See Ouster's [point definition](https://github.com/ouster-lidar/ouster-ros/blob/ros2/ouster-ros/include/ouster_ros/os_point.h)
and [sensor data documentation](https://static.ouster.dev/sensor-docs/image_route1/image_route3/sensor_data/sensor-data.html).

## Selecting a channel

For the **3D photometric residual**, set:

```yaml
odom/gicp/photometricChannel: reflectivity
odom/gicp/photometricScale: 255.0
odom/gicp/photometricWeight: 0.1  # example trial value; validate on your data
```

For raw signal instead, use `photometricChannel: intensity` (`signal` is an
alias). Choose `photometricScale` for the signal range; `65535.0` is appropriate
for a 16-bit signal profile. DLIIO preserves `intensity` and computes
`intensity_corrected = intensity * (range / intensityRRef)^intensityAlpha`, with
optional incidence correction. There is no upper display clamp. Set
`intensityAlpha: 0.0` and leave incidence correction off to use raw signal
without the range model. The model and reference range require sensor-specific
validation; normalization does not make weights universally transferable.

For the **LiDAR-image residuals**, `odom/lidar_image/channel` independently selects
`reflectivity`, `intensity`/`signal`, or `ambient`/`near_ir`.
`odom/lidar_image/scale` normalizes this image channel. Both frame-to-map and
flow-only configurations honor selection and `denoiseKernel`; these operations
never overwrite calibrated reflectivity or raw intensity. The selected values
are carried in `lidar_intensity`, including the voxel-based reference modes.
Full-resolution keyframe-image references (`imageRefs: true`) and flow image
references (`flow/imageRef: true`) avoid voxel averaging on the reference side.

The image terms require organized scans and the existing spherical projection
model. Projection accounts for azimuth wrapping across the 360-degree image;
keyframe references, map residuals, and flow use the same convention.
Field decoding alone does not validate that model for a particular
sensor's beam offsets, scan organization, or extrinsics.

Photometric channel fallback is resolved on the first nonempty cloud, even if
the weight is initially zero. If neither channel is readable, the term stays
disabled when a live weight change is requested. An unavailable image channel
falls back to readable raw intensity; with no fallback the image is skipped.
The configured scale is retained during fallback, so check it against the actual
field reported in diagnostics. Restart after changing the input point format or
startup-only channel/preprocessing parameters.

## Verifying engagement

`/diagnostics` reports `Input Intensity Field`, `Input Reflectivity Field`,
`Photometric Channel`, and `Lidar Image Channel` (the actual field after aliases
and fallback). Check residual counts as well: `Photometric Points`,
`Lidar Map Points`, and `Lidar Flow Points` establish that a term contributed;
an enabled parameter alone does not establish that usable texture was found.

`test_pointcloud_channels` exercises the node's actual intake, deskew, and voxel
path with binary Ouster original/native layouts, row padding, separate channels,
flow-only denoising, missing fields, and live enabling. Registration tests cover
corrected intensity, gradient refresh after live changes, and independent
image-channel references in both map and flow residuals.

The internal aligned point type grows from 32 to 48 bytes to retain raw and
derived channels independently. Rebuild all consumers of the C++ headers.
Synthetic regression tests and field checks do not establish improved trajectory
accuracy; that requires repeatable bag comparisons with appropriate ground truth.
