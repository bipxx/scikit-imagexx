# API implementation inventory

Reference: scikit-image 0.26.0.

Phase-2 source registers 236 partial image APIs plus 3 metadata functions and 5 partial native classes. Recent increments add owned entropy, block sums, perimeter/Crofton, Euler, polygon and mesh arithmetic plus profile-line sampling, blur arithmetic and image comparisons alongside the existing filters and moment kernels. Per-function contracts, measured workloads and retained primitive-provider partitions are recorded in [the fast-run guide](FAST_RUN.md). Combined supported-profile wheel/SDK qualification remains at 203 image APIs; source counts, component checks and scoped performance results are separate from that qualification.

Qualified product: `ea35a2481880301b8ce7b4eb2d77665adb813ca2`, Actions run `40237475475`: four profiles, 828,304 XML checks with no failures/errors/skips, three Release products. The qualified composition includes reconstruction, normalized mutual information, H-extrema, the metrics SIMD increment and the bounded TV zero-scan change. This is partial per-function scope, not full API parity or performance acceptance.

477 public callable names inventoried in 20 modules; 236 have partially native image implementations and 3 have metadata-only execution. 5 have separate partial native class/state implementations and 1 have exception metadata providers; these never increase image-kernel counts. Covered entries link to per-function test and benchmark files; these paths do not prove a passing contract or measured performance.

An inventoried name is not evidence of parity. `upstream-only` entries have no native implementation, no dedicated parity test, and no native performance result. Modules outside this inventory remain unassessed. Native entries support a documented subset of input/options; unsupported calls can use explicit upstream fallback. Metadata entries return small values/arrays without running an image kernel. Identity-preserving conversion calls can return the input without processing pixels.

| API | Implementation | Dedicated parity test file | Benchmark file |
|---|---|---|---|
| `color.combine_stains` | native-partial | yes (partial input contract) | yes |
| `color.convert_colorspace` | native-partial | yes (partial input contract) | yes |
| `color.deltaE_cie76` | native-partial | yes (partial input contract) | yes |
| `color.deltaE_ciede2000` | native-partial | yes (partial input contract) | yes |
| `color.deltaE_ciede94` | native-partial | yes (partial input contract) | yes |
| `color.deltaE_cmc` | native-partial | yes (partial input contract) | yes |
| `color.gray2rgb` | native-partial | yes (partial input contract) | yes |
| `color.gray2rgba` | native-partial | yes (partial input contract) | yes |
| `color.hed2rgb` | native-partial | yes (partial input contract) | yes |
| `color.hsv2rgb` | native-partial | yes (partial input contract) | yes |
| `color.lab2lch` | native-partial | yes (partial input contract) | yes |
| `color.lab2rgb` | native-partial | yes (partial input contract) | yes |
| `color.lab2xyz` | native-partial | yes (partial input contract) | yes |
| `color.label2rgb` | native-partial | yes (partial input contract) | yes |
| `color.lch2lab` | native-partial | yes (partial input contract) | yes |
| `color.luv2rgb` | native-partial | yes (partial input contract) | yes |
| `color.luv2xyz` | native-partial | yes (partial input contract) | yes |
| `color.rgb2gray` | native-partial | yes (partial input contract) | yes |
| `color.rgb2hed` | native-partial | yes (partial input contract) | yes |
| `color.rgb2hsv` | native-partial | yes (partial input contract) | yes |
| `color.rgb2lab` | native-partial | yes (partial input contract) | yes |
| `color.rgb2luv` | native-partial | yes (partial input contract) | yes |
| `color.rgb2rgbcie` | native-partial | yes (partial input contract) | yes |
| `color.rgb2xyz` | native-partial | yes (partial input contract) | yes |
| `color.rgb2ycbcr` | native-partial | yes (partial input contract) | yes |
| `color.rgb2ydbdr` | native-partial | yes (partial input contract) | yes |
| `color.rgb2yiq` | native-partial | yes (partial input contract) | yes |
| `color.rgb2ypbpr` | native-partial | yes (partial input contract) | yes |
| `color.rgb2yuv` | native-partial | yes (partial input contract) | yes |
| `color.rgba2rgb` | native-partial | yes (partial input contract) | yes |
| `color.rgbcie2rgb` | native-partial | yes (partial input contract) | yes |
| `color.separate_stains` | native-partial | yes (partial input contract) | yes |
| `color.xyz2lab` | native-partial | yes (partial input contract) | yes |
| `color.xyz2luv` | native-partial | yes (partial input contract) | yes |
| `color.xyz2rgb` | native-partial | yes (partial input contract) | yes |
| `color.xyz_tristimulus_values` | metadata | yes (partial input contract) | yes |
| `color.ycbcr2rgb` | native-partial | yes (partial input contract) | yes |
| `color.ydbdr2rgb` | native-partial | yes (partial input contract) | yes |
| `color.yiq2rgb` | native-partial | yes (partial input contract) | yes |
| `color.ypbpr2rgb` | native-partial | yes (partial input contract) | yes |
| `color.yuv2rgb` | native-partial | yes (partial input contract) | yes |
| `data.astronaut` | upstream-only | pending | pending |
| `data.binary_blobs` | upstream-only | pending | pending |
| `data.brain` | upstream-only | pending | pending |
| `data.brick` | upstream-only | pending | pending |
| `data.camera` | upstream-only | pending | pending |
| `data.cat` | upstream-only | pending | pending |
| `data.cell` | upstream-only | pending | pending |
| `data.cells3d` | upstream-only | pending | pending |
| `data.checkerboard` | upstream-only | pending | pending |
| `data.chelsea` | upstream-only | pending | pending |
| `data.clock` | upstream-only | pending | pending |
| `data.coffee` | upstream-only | pending | pending |
| `data.coins` | upstream-only | pending | pending |
| `data.colorwheel` | upstream-only | pending | pending |
| `data.download_all` | upstream-only | pending | pending |
| `data.eagle` | upstream-only | pending | pending |
| `data.file_hash` | upstream-only | pending | pending |
| `data.grass` | upstream-only | pending | pending |
| `data.gravel` | upstream-only | pending | pending |
| `data.horse` | upstream-only | pending | pending |
| `data.hubble_deep_field` | upstream-only | pending | pending |
| `data.human_mitosis` | upstream-only | pending | pending |
| `data.immunohistochemistry` | upstream-only | pending | pending |
| `data.kidney` | upstream-only | pending | pending |
| `data.lbp_frontal_face_cascade_filename` | upstream-only | pending | pending |
| `data.lfw_subset` | upstream-only | pending | pending |
| `data.lily` | upstream-only | pending | pending |
| `data.logo` | upstream-only | pending | pending |
| `data.microaneurysms` | upstream-only | pending | pending |
| `data.moon` | upstream-only | pending | pending |
| `data.nickel_solidification` | upstream-only | pending | pending |
| `data.page` | upstream-only | pending | pending |
| `data.palisades_of_vogt` | upstream-only | pending | pending |
| `data.protein_transport` | upstream-only | pending | pending |
| `data.retina` | upstream-only | pending | pending |
| `data.rocket` | upstream-only | pending | pending |
| `data.shepp_logan_phantom` | upstream-only | pending | pending |
| `data.skin` | upstream-only | pending | pending |
| `data.stereo_motorcycle` | upstream-only | pending | pending |
| `data.text` | upstream-only | pending | pending |
| `data.vortex` | upstream-only | pending | pending |
| `draw.bezier_curve` | upstream-only | pending | pending |
| `draw.circle_perimeter` | upstream-only | pending | pending |
| `draw.circle_perimeter_aa` | upstream-only | pending | pending |
| `draw.disk` | upstream-only | pending | pending |
| `draw.ellipse` | upstream-only | pending | pending |
| `draw.ellipse_perimeter` | upstream-only | pending | pending |
| `draw.ellipsoid` | upstream-only | pending | pending |
| `draw.ellipsoid_stats` | upstream-only | pending | pending |
| `draw.line` | upstream-only | pending | pending |
| `draw.line_aa` | upstream-only | pending | pending |
| `draw.line_nd` | upstream-only | pending | pending |
| `draw.polygon` | upstream-only | pending | pending |
| `draw.polygon2mask` | upstream-only | pending | pending |
| `draw.polygon_perimeter` | upstream-only | pending | pending |
| `draw.random_shapes` | upstream-only | pending | pending |
| `draw.rectangle` | upstream-only | pending | pending |
| `draw.rectangle_perimeter` | upstream-only | pending | pending |
| `draw.set_color` | upstream-only | pending | pending |
| `exposure.adjust_gamma` | native-partial | yes (partial input contract) | yes |
| `exposure.adjust_log` | native-partial | yes (partial input contract) | yes |
| `exposure.adjust_sigmoid` | native-partial | yes (partial input contract) | yes |
| `exposure.cumulative_distribution` | native-partial | yes (partial input contract) | yes |
| `exposure.equalize_adapthist` | native-partial | yes (partial input contract) | yes |
| `exposure.equalize_hist` | native-partial | yes (partial input contract) | yes |
| `exposure.histogram` | native-partial | yes (partial input contract) | yes |
| `exposure.is_low_contrast` | native-partial | yes (partial input contract) | yes |
| `exposure.match_histograms` | native-partial | yes (partial input contract) | yes |
| `exposure.rescale_intensity` | native-partial | yes (partial input contract) | yes |
| `feature.BRIEF` | upstream-only | pending | pending |
| `feature.CENSURE` | upstream-only | pending | pending |
| `feature.Cascade` | upstream-only | pending | pending |
| `feature.ORB` | upstream-only | pending | pending |
| `feature.SIFT` | upstream-only | pending | pending |
| `feature.blob_dog` | upstream-only | pending | pending |
| `feature.blob_doh` | upstream-only | pending | pending |
| `feature.blob_log` | upstream-only | pending | pending |
| `feature.canny` | upstream-only | pending | pending |
| `feature.corner_fast` | upstream-only | pending | pending |
| `feature.corner_foerstner` | native-partial | yes (partial input contract) | yes |
| `feature.corner_harris` | native-partial | yes (partial input contract) | yes |
| `feature.corner_kitchen_rosenfeld` | native-partial | yes (partial input contract) | yes |
| `feature.corner_moravec` | upstream-only | pending | pending |
| `feature.corner_orientations` | upstream-only | pending | pending |
| `feature.corner_peaks` | upstream-only | pending | pending |
| `feature.corner_shi_tomasi` | native-partial | yes (partial input contract) | yes |
| `feature.corner_subpix` | upstream-only | pending | pending |
| `feature.daisy` | upstream-only | pending | pending |
| `feature.draw_haar_like_feature` | upstream-only | pending | pending |
| `feature.draw_multiblock_lbp` | upstream-only | pending | pending |
| `feature.fisher_vector` | upstream-only | pending | pending |
| `feature.graycomatrix` | upstream-only | pending | pending |
| `feature.graycoprops` | upstream-only | pending | pending |
| `feature.haar_like_feature` | upstream-only | pending | pending |
| `feature.haar_like_feature_coord` | upstream-only | pending | pending |
| `feature.hessian_matrix` | native-partial | yes (partial input contract) | yes |
| `feature.hessian_matrix_det` | native-partial | yes (partial input contract) | yes |
| `feature.hessian_matrix_eigvals` | native-partial | yes (partial input contract) | yes |
| `feature.hog` | upstream-only | pending | pending |
| `feature.learn_gmm` | upstream-only | pending | pending |
| `feature.local_binary_pattern` | upstream-only | pending | pending |
| `feature.match_descriptors` | upstream-only | pending | pending |
| `feature.match_template` | upstream-only | pending | pending |
| `feature.multiblock_lbp` | upstream-only | pending | pending |
| `feature.multiscale_basic_features` | upstream-only | pending | pending |
| `feature.peak_local_max` | upstream-only | pending | pending |
| `feature.plot_matched_features` | upstream-only | pending | pending |
| `feature.shape_index` | native-partial | yes (partial input contract) | yes |
| `feature.structure_tensor` | native-partial | yes (partial input contract) | yes |
| `feature.structure_tensor_eigenvalues` | native-partial | yes (partial input contract) | yes |
| `filters.LPIFilter2D` | native-class-partial | yes (partial input contract) | pending |
| `filters.apply_hysteresis_threshold` | native-partial | yes (partial input contract) | yes |
| `filters.butterworth` | native-partial | yes (partial input contract) | yes |
| `filters.correlate_sparse` | native-partial | yes (partial input contract) | yes |
| `filters.difference_of_gaussians` | native-partial | yes (partial input contract) | yes |
| `filters.farid` | native-partial | yes (partial input contract) | yes |
| `filters.farid_h` | native-partial | yes (partial input contract) | yes |
| `filters.farid_v` | native-partial | yes (partial input contract) | yes |
| `filters.filter_forward` | native-partial | yes (partial input contract) | yes |
| `filters.filter_inverse` | native-partial | yes (partial input contract) | yes |
| `filters.frangi` | native-partial | yes (partial input contract) | yes |
| `filters.gabor` | native-partial | yes (partial input contract) | yes |
| `filters.gabor_kernel` | native-partial | yes (partial input contract) | yes |
| `filters.gaussian` | native-partial | yes (partial input contract) | yes |
| `filters.hessian` | native-partial | yes (partial input contract) | yes |
| `filters.laplace` | native-partial | yes (partial input contract) | yes |
| `filters.median` | native-partial | yes (partial input contract) | yes |
| `filters.meijering` | native-partial | yes (partial input contract) | yes |
| `filters.prewitt` | native-partial | yes (partial input contract) | yes |
| `filters.prewitt_h` | native-partial | yes (partial input contract) | yes |
| `filters.prewitt_v` | native-partial | yes (partial input contract) | yes |
| `filters.rank_order` | native-partial | yes (partial input contract) | yes |
| `filters.roberts` | native-partial | yes (partial input contract) | yes |
| `filters.roberts_neg_diag` | native-partial | yes (partial input contract) | yes |
| `filters.roberts_pos_diag` | native-partial | yes (partial input contract) | yes |
| `filters.sato` | native-partial | yes (partial input contract) | yes |
| `filters.scharr` | native-partial | yes (partial input contract) | yes |
| `filters.scharr_h` | native-partial | yes (partial input contract) | yes |
| `filters.scharr_v` | native-partial | yes (partial input contract) | yes |
| `filters.sobel` | native-partial | yes (partial input contract) | yes |
| `filters.sobel_h` | native-partial | yes (partial input contract) | yes |
| `filters.sobel_v` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_isodata` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_li` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_local` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_mean` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_minimum` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_multiotsu` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_niblack` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_otsu` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_sauvola` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_triangle` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_yen` | native-partial | yes (partial input contract) | yes |
| `filters.try_all_threshold` | upstream-only | pending | pending |
| `filters.unsharp_mask` | native-partial | yes (partial input contract) | yes |
| `filters.wiener` | native-partial | yes (partial input contract) | yes |
| `filters.window` | native-partial | yes (partial input contract) | yes |
| `filters.rank.autolevel` | native-partial | yes (partial input contract) | yes |
| `filters.rank.autolevel_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.enhance_contrast` | native-partial | yes (partial input contract) | yes |
| `filters.rank.enhance_contrast_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.entropy` | native-partial | yes (partial input contract) | yes |
| `filters.rank.equalize` | native-partial | yes (partial input contract) | yes |
| `filters.rank.geometric_mean` | native-partial | yes (partial input contract) | yes |
| `filters.rank.gradient` | native-partial | yes (partial input contract) | yes |
| `filters.rank.gradient_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.majority` | native-partial | yes (partial input contract) | yes |
| `filters.rank.maximum` | native-partial | yes (partial input contract) | yes |
| `filters.rank.mean` | native-partial | yes (partial input contract) | yes |
| `filters.rank.mean_bilateral` | native-partial | yes (partial input contract) | yes |
| `filters.rank.mean_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.median` | native-partial | yes (partial input contract) | yes |
| `filters.rank.minimum` | native-partial | yes (partial input contract) | yes |
| `filters.rank.modal` | native-partial | yes (partial input contract) | yes |
| `filters.rank.noise_filter` | native-partial | yes (partial input contract) | yes |
| `filters.rank.otsu` | native-partial | yes (partial input contract) | yes |
| `filters.rank.percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.pop` | native-partial | yes (partial input contract) | yes |
| `filters.rank.pop_bilateral` | native-partial | yes (partial input contract) | yes |
| `filters.rank.pop_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.subtract_mean` | native-partial | yes (partial input contract) | yes |
| `filters.rank.subtract_mean_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.sum` | native-partial | yes (partial input contract) | yes |
| `filters.rank.sum_bilateral` | native-partial | yes (partial input contract) | yes |
| `filters.rank.sum_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.threshold` | native-partial | yes (partial input contract) | yes |
| `filters.rank.threshold_percentile` | native-partial | yes (partial input contract) | yes |
| `filters.rank.windowed_histogram` | native-partial | yes (partial input contract) | yes |
| `filters.rank.bilateral.mean_bilateral` | native-partial | yes (partial input contract) | yes |
| `filters.rank.bilateral.pop_bilateral` | native-partial | yes (partial input contract) | yes |
| `filters.rank.bilateral.sum_bilateral` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.autolevel` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.enhance_contrast` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.entropy` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.equalize` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.geometric_mean` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.gradient` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.majority` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.maximum` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.mean` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.median` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.minimum` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.modal` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.noise_filter` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.otsu` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.pop` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.subtract_mean` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.sum` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.threshold` | native-partial | yes (partial input contract) | yes |
| `filters.rank.generic.windowed_histogram` | native-partial | yes (partial input contract) | yes |
| `future.TrainableSegmenter` | upstream-only | pending | pending |
| `future.fit_segmenter` | upstream-only | pending | pending |
| `future.manual_lasso_segmentation` | upstream-only | pending | pending |
| `future.manual_polygon_segmentation` | upstream-only | pending | pending |
| `future.predict_segmenter` | upstream-only | pending | pending |
| `graph.MCP` | upstream-only | pending | pending |
| `graph.MCP_Connect` | upstream-only | pending | pending |
| `graph.MCP_Flexible` | upstream-only | pending | pending |
| `graph.MCP_Geometric` | upstream-only | pending | pending |
| `graph.RAG` | upstream-only | pending | pending |
| `graph.central_pixel` | upstream-only | pending | pending |
| `graph.cut_normalized` | upstream-only | pending | pending |
| `graph.cut_threshold` | upstream-only | pending | pending |
| `graph.merge_hierarchical` | upstream-only | pending | pending |
| `graph.pixel_graph` | upstream-only | pending | pending |
| `graph.rag_boundary` | upstream-only | pending | pending |
| `graph.rag_mean_color` | upstream-only | pending | pending |
| `graph.route_through_array` | upstream-only | pending | pending |
| `graph.shortest_path` | upstream-only | pending | pending |
| `graph.show_rag` | upstream-only | pending | pending |
| `io.ImageCollection` | upstream-only | pending | pending |
| `io.MultiImage` | upstream-only | pending | pending |
| `io.concatenate_images` | upstream-only | pending | pending |
| `io.imread` | upstream-only | pending | pending |
| `io.imread_collection` | upstream-only | pending | pending |
| `io.imread_collection_wrapper` | upstream-only | pending | pending |
| `io.imsave` | upstream-only | pending | pending |
| `io.load_sift` | upstream-only | pending | pending |
| `io.load_surf` | upstream-only | pending | pending |
| `io.pop` | upstream-only | pending | pending |
| `io.push` | upstream-only | pending | pending |
| `measure.CircleModel` | upstream-only | pending | pending |
| `measure.EllipseModel` | upstream-only | pending | pending |
| `measure.LineModelND` | upstream-only | pending | pending |
| `measure.RansacModelProtocol` | upstream-only | pending | pending |
| `measure.approximate_polygon` | upstream-only | pending | pending |
| `measure.block_reduce` | upstream-only | pending | pending |
| `measure.blur_effect` | native-partial | yes (partial input contract) | yes |
| `measure.centroid` | native-partial | yes (partial input contract) | yes |
| `measure.euler_number` | upstream-only | pending | pending |
| `measure.find_contours` | upstream-only | pending | pending |
| `measure.grid_points_in_poly` | upstream-only | pending | pending |
| `measure.inertia_tensor` | upstream-only | pending | pending |
| `measure.inertia_tensor_eigvals` | upstream-only | pending | pending |
| `measure.intersection_coeff` | native-partial | yes (partial input contract) | yes |
| `measure.label` | native-partial | yes (partial input contract) | yes |
| `measure.manders_coloc_coeff` | native-partial | yes (partial input contract) | yes |
| `measure.manders_overlap_coeff` | native-partial | yes (partial input contract) | yes |
| `measure.marching_cubes` | upstream-only | pending | pending |
| `measure.mesh_surface_area` | upstream-only | pending | pending |
| `measure.moments` | native-partial | yes (partial input contract) | yes |
| `measure.moments_central` | native-partial | yes (partial input contract) | yes |
| `measure.moments_coords` | upstream-only | pending | pending |
| `measure.moments_coords_central` | upstream-only | pending | pending |
| `measure.moments_hu` | native-partial | yes (partial input contract) | yes |
| `measure.moments_normalized` | native-partial | yes (partial input contract) | yes |
| `measure.pearson_corr_coeff` | upstream-only | pending | pending |
| `measure.perimeter` | upstream-only | pending | pending |
| `measure.perimeter_crofton` | upstream-only | pending | pending |
| `measure.points_in_poly` | upstream-only | pending | pending |
| `measure.profile_line` | native-partial | yes (partial input contract) | yes |
| `measure.ransac` | upstream-only | pending | pending |
| `measure.regionprops` | upstream-only | pending | pending |
| `measure.regionprops_table` | upstream-only | pending | pending |
| `measure.shannon_entropy` | upstream-only | pending | pending |
| `measure.subdivide_polygon` | upstream-only | pending | pending |
| `metrics.adapted_rand_error` | upstream-only | pending | pending |
| `metrics.contingency_table` | upstream-only | pending | pending |
| `metrics.hausdorff_distance` | upstream-only | pending | pending |
| `metrics.hausdorff_pair` | upstream-only | pending | pending |
| `metrics.mean_squared_error` | native-partial | yes (partial input contract) | yes |
| `metrics.normalized_mutual_information` | native-partial | yes (partial input contract) | yes |
| `metrics.normalized_root_mse` | native-partial | yes (partial input contract) | yes |
| `metrics.peak_signal_noise_ratio` | native-partial | yes (partial input contract) | yes |
| `metrics.structural_similarity` | upstream-only | pending | pending |
| `metrics.variation_of_information` | upstream-only | pending | pending |
| `morphology.area_closing` | native-partial | yes (partial input contract) | yes |
| `morphology.area_opening` | native-partial | yes (partial input contract) | yes |
| `morphology.ball` | native-partial | yes (partial input contract) | yes |
| `morphology.binary_closing` | native-partial | yes (partial input contract) | yes |
| `morphology.binary_dilation` | native-partial | yes (partial input contract) | yes |
| `morphology.binary_erosion` | native-partial | yes (partial input contract) | yes |
| `morphology.binary_opening` | native-partial | yes (partial input contract) | yes |
| `morphology.black_tophat` | native-partial | yes (partial input contract) | yes |
| `morphology.closing` | native-partial | yes (partial input contract) | yes |
| `morphology.convex_hull_image` | upstream-only | pending | pending |
| `morphology.convex_hull_object` | upstream-only | pending | pending |
| `morphology.diameter_closing` | native-partial | yes (partial input contract) | yes |
| `morphology.diameter_opening` | native-partial | yes (partial input contract) | yes |
| `morphology.diamond` | native-partial | yes (partial input contract) | yes |
| `morphology.dilation` | native-partial | yes (partial input contract) | yes |
| `morphology.disk` | native-partial | yes (partial input contract) | yes |
| `morphology.ellipse` | native-partial | yes (partial input contract) | yes |
| `morphology.erosion` | native-partial | yes (partial input contract) | yes |
| `morphology.flood` | upstream-only | pending | pending |
| `morphology.flood_fill` | upstream-only | pending | pending |
| `morphology.footprint_from_sequence` | native-partial | yes (partial input contract) | yes |
| `morphology.footprint_rectangle` | native-partial | yes (partial input contract) | yes |
| `morphology.h_maxima` | native-partial | yes (partial input contract) | yes |
| `morphology.h_minima` | native-partial | yes (partial input contract) | yes |
| `morphology.isotropic_closing` | upstream-only | pending | pending |
| `morphology.isotropic_dilation` | upstream-only | pending | pending |
| `morphology.isotropic_erosion` | upstream-only | pending | pending |
| `morphology.isotropic_opening` | upstream-only | pending | pending |
| `morphology.label` | native-partial | yes (partial input contract) | yes |
| `morphology.local_maxima` | upstream-only | pending | pending |
| `morphology.local_minima` | upstream-only | pending | pending |
| `morphology.max_tree` | native-partial | yes (partial input contract) | yes |
| `morphology.max_tree_local_maxima` | native-partial | yes (partial input contract) | yes |
| `morphology.medial_axis` | upstream-only | pending | pending |
| `morphology.mirror_footprint` | metadata | yes (partial input contract) | yes |
| `morphology.octagon` | native-partial | yes (partial input contract) | yes |
| `morphology.octahedron` | native-partial | yes (partial input contract) | yes |
| `morphology.opening` | native-partial | yes (partial input contract) | yes |
| `morphology.pad_footprint` | native-partial | yes (partial input contract) | yes |
| `morphology.reconstruction` | native-partial | yes (partial input contract) | yes |
| `morphology.remove_objects_by_distance` | upstream-only | pending | pending |
| `morphology.remove_small_holes` | native-partial | yes (partial input contract) | yes |
| `morphology.remove_small_objects` | native-partial | yes (partial input contract) | yes |
| `morphology.skeletonize` | upstream-only | pending | pending |
| `morphology.star` | native-partial | yes (partial input contract) | yes |
| `morphology.thin` | upstream-only | pending | pending |
| `morphology.white_tophat` | native-partial | yes (partial input contract) | yes |
| `registration.optical_flow_ilk` | upstream-only | pending | pending |
| `registration.optical_flow_tvl1` | upstream-only | pending | pending |
| `registration.phase_cross_correlation` | upstream-only | pending | pending |
| `restoration.ball_kernel` | upstream-only | pending | pending |
| `restoration.calibrate_denoiser` | upstream-only | pending | pending |
| `restoration.cycle_spin` | upstream-only | pending | pending |
| `restoration.denoise_bilateral` | upstream-only | pending | pending |
| `restoration.denoise_invariant` | upstream-only | pending | pending |
| `restoration.denoise_nl_means` | upstream-only | pending | pending |
| `restoration.denoise_tv_bregman` | native-partial | yes (partial input contract) | yes |
| `restoration.denoise_tv_chambolle` | native-partial | yes (partial input contract) | yes |
| `restoration.denoise_wavelet` | upstream-only | pending | pending |
| `restoration.ellipsoid_kernel` | upstream-only | pending | pending |
| `restoration.estimate_sigma` | upstream-only | pending | pending |
| `restoration.inpaint_biharmonic` | upstream-only | pending | pending |
| `restoration.richardson_lucy` | upstream-only | pending | pending |
| `restoration.rolling_ball` | upstream-only | pending | pending |
| `restoration.unsupervised_wiener` | upstream-only | pending | pending |
| `restoration.unwrap_phase` | upstream-only | pending | pending |
| `restoration.wiener` | upstream-only | pending | pending |
| `segmentation.active_contour` | upstream-only | pending | pending |
| `segmentation.chan_vese` | upstream-only | pending | pending |
| `segmentation.checkerboard_level_set` | upstream-only | pending | pending |
| `segmentation.clear_border` | native-partial | yes (partial input contract) | yes |
| `segmentation.disk_level_set` | upstream-only | pending | pending |
| `segmentation.expand_labels` | upstream-only | pending | pending |
| `segmentation.felzenszwalb` | upstream-only | pending | pending |
| `segmentation.find_boundaries` | upstream-only | pending | pending |
| `segmentation.flood` | upstream-only | pending | pending |
| `segmentation.flood_fill` | upstream-only | pending | pending |
| `segmentation.inverse_gaussian_gradient` | upstream-only | pending | pending |
| `segmentation.join_segmentations` | upstream-only | pending | pending |
| `segmentation.mark_boundaries` | upstream-only | pending | pending |
| `segmentation.morphological_chan_vese` | upstream-only | pending | pending |
| `segmentation.morphological_geodesic_active_contour` | upstream-only | pending | pending |
| `segmentation.quickshift` | upstream-only | pending | pending |
| `segmentation.random_walker` | upstream-only | pending | pending |
| `segmentation.relabel_sequential` | native-partial | yes (partial input contract) | yes |
| `segmentation.slic` | upstream-only | pending | pending |
| `segmentation.watershed` | upstream-only | pending | pending |
| `transform.AffineTransform` | native-class-partial | yes (partial input contract) | pending |
| `transform.EssentialMatrixTransform` | upstream-only | pending | pending |
| `transform.EuclideanTransform` | native-class-partial | yes (partial input contract) | pending |
| `transform.FundamentalMatrixTransform` | upstream-only | pending | pending |
| `transform.PiecewiseAffineTransform` | upstream-only | pending | pending |
| `transform.PolynomialTransform` | upstream-only | pending | pending |
| `transform.ProjectiveTransform` | native-class-partial | yes (partial input contract) | pending |
| `transform.SimilarityTransform` | native-class-partial | yes (partial input contract) | pending |
| `transform.ThinPlateSplineTransform` | upstream-only | pending | pending |
| `transform.downscale_local_mean` | native-partial | yes (partial input contract) | yes |
| `transform.estimate_transform` | upstream-only | pending | pending |
| `transform.frt2` | upstream-only | pending | pending |
| `transform.hough_circle` | upstream-only | pending | pending |
| `transform.hough_circle_peaks` | upstream-only | pending | pending |
| `transform.hough_ellipse` | upstream-only | pending | pending |
| `transform.hough_line` | upstream-only | pending | pending |
| `transform.hough_line_peaks` | upstream-only | pending | pending |
| `transform.ifrt2` | upstream-only | pending | pending |
| `transform.integral_image` | native-partial | yes (partial input contract) | yes |
| `transform.integrate` | native-partial | yes (partial input contract) | yes |
| `transform.iradon` | upstream-only | pending | pending |
| `transform.iradon_sart` | upstream-only | pending | pending |
| `transform.matrix_transform` | native-partial | yes (partial input contract) | yes |
| `transform.order_angles_golden_ratio` | upstream-only | pending | pending |
| `transform.probabilistic_hough_line` | upstream-only | pending | pending |
| `transform.pyramid_expand` | upstream-only | pending | pending |
| `transform.pyramid_gaussian` | upstream-only | pending | pending |
| `transform.pyramid_laplacian` | upstream-only | pending | pending |
| `transform.pyramid_reduce` | upstream-only | pending | pending |
| `transform.radon` | upstream-only | pending | pending |
| `transform.rescale` | native-partial | yes (partial input contract) | yes |
| `transform.resize` | native-partial | yes (partial input contract) | yes |
| `transform.resize_local_mean` | upstream-only | pending | pending |
| `transform.rotate` | upstream-only | pending | pending |
| `transform.swirl` | upstream-only | pending | pending |
| `transform.warp` | upstream-only | pending | pending |
| `transform.warp_coords` | upstream-only | pending | pending |
| `transform.warp_polar` | upstream-only | pending | pending |
| `util.FailedEstimationAccessError` | exception-metadata-provider | yes (partial input contract) | pending |
| `util.PendingSkimage2Change` | upstream-only | pending | pending |
| `util.apply_parallel` | upstream-only | pending | pending |
| `util.compare_images` | native-partial | yes (partial input contract) | yes |
| `util.crop` | upstream-only | pending | pending |
| `util.dtype_limits` | metadata | yes (partial input contract) | yes |
| `util.img_as_bool` | native-partial | yes (partial input contract) | yes |
| `util.img_as_float` | native-partial | yes (partial input contract) | yes |
| `util.img_as_float32` | native-partial | yes (partial input contract) | yes |
| `util.img_as_float64` | native-partial | yes (partial input contract) | yes |
| `util.img_as_int` | native-partial | yes (partial input contract) | yes |
| `util.img_as_ubyte` | native-partial | yes (partial input contract) | yes |
| `util.img_as_uint` | native-partial | yes (partial input contract) | yes |
| `util.invert` | native-partial | yes (partial input contract) | yes |
| `util.label_points` | upstream-only | pending | pending |
| `util.lookfor` | upstream-only | pending | pending |
| `util.map_array` | upstream-only | pending | pending |
| `util.montage` | upstream-only | pending | pending |
| `util.random_noise` | upstream-only | pending | pending |
| `util.regular_grid` | upstream-only | pending | pending |
| `util.regular_seeds` | upstream-only | pending | pending |
| `util.slice_along_axes` | upstream-only | pending | pending |
| `util.unique_rows` | upstream-only | pending | pending |
| `util.view_as_blocks` | upstream-only | pending | pending |
| `util.view_as_windows` | upstream-only | pending | pending |

## Class member inventory

29 exported class names expose 284 statically discoverable member paths, including inherited methods and properties. Their signatures and defining classes are recorded in `api.json`. Registered class/member implementations and provider scope are separate from passing parity or measured performance. Unregistered members still require implementation, numerical/state parity and benchmarks; none increase image coverage. Constructors are not executed and property getters are not evaluated during inventory. Instance fields and runtime-generated attributes remain unassessed.

| Class | Static member paths | Native member parity |
|---|---:|---|
| `feature.BRIEF` | 2 | pending |
| `feature.CENSURE` | 2 | pending |
| `feature.Cascade` | 11 | pending |
| `feature.ORB` | 4 | pending |
| `feature.SIFT` | 5 | pending |
| `filters.LPIFilter2D` | 2 | partial implementation; evidence separate |
| `future.TrainableSegmenter` | 4 | pending |
| `graph.MCP` | 10 | pending |
| `graph.MCP_Connect` | 11 | pending |
| `graph.MCP_Flexible` | 13 | pending |
| `graph.MCP_Geometric` | 10 | pending |
| `graph.RAG` | 46 | pending |
| `io.ImageCollection` | 9 | pending |
| `io.MultiImage` | 10 | pending |
| `measure.CircleModel` | 6 | pending |
| `measure.EllipseModel` | 6 | pending |
| `measure.LineModelND` | 8 | pending |
| `measure.RansacModelProtocol` | 6 | pending |
| `transform.AffineTransform` | 17 | partial implementation; evidence separate |
| `transform.EssentialMatrixTransform` | 8 | pending |
| `transform.EuclideanTransform` | 15 | partial implementation; evidence separate |
| `transform.FundamentalMatrixTransform` | 8 | pending |
| `transform.PiecewiseAffineTransform` | 7 | pending |
| `transform.PolynomialTransform` | 7 | pending |
| `transform.ProjectiveTransform` | 13 | partial implementation; evidence separate |
| `transform.SimilarityTransform` | 16 | partial implementation; evidence separate |
| `transform.ThinPlateSplineTransform` | 5 | pending |
| `util.FailedEstimationAccessError` | 13 | pending |
| `util.PendingSkimage2Change` | 10 | pending |
