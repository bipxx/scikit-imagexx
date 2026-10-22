# API implementation inventory

Reference: scikit-image 0.26.0.

455 public callable names inventoried in 18 modules; 99 have partially native image implementations and 2 have metadata-only execution. All covered entries have per-function tests and benchmarks.

An inventoried name is not evidence of parity. `upstream-only` entries have no native implementation, no dedicated parity test, and no native performance result. Modules outside this inventory remain unassessed. Native entries support a documented subset of input/options; unsupported calls can use explicit upstream fallback. Metadata entries return small values/arrays without running an image kernel. Identity-preserving conversion calls can return the input without processing pixels.

| API | Implementation | Dedicated parity test | Benchmark |
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
| `exposure.equalize_adapthist` | upstream-only | pending | pending |
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
| `feature.corner_foerstner` | upstream-only | pending | pending |
| `feature.corner_harris` | upstream-only | pending | pending |
| `feature.corner_kitchen_rosenfeld` | upstream-only | pending | pending |
| `feature.corner_moravec` | upstream-only | pending | pending |
| `feature.corner_orientations` | upstream-only | pending | pending |
| `feature.corner_peaks` | upstream-only | pending | pending |
| `feature.corner_shi_tomasi` | upstream-only | pending | pending |
| `feature.corner_subpix` | upstream-only | pending | pending |
| `feature.daisy` | upstream-only | pending | pending |
| `feature.draw_haar_like_feature` | upstream-only | pending | pending |
| `feature.draw_multiblock_lbp` | upstream-only | pending | pending |
| `feature.fisher_vector` | upstream-only | pending | pending |
| `feature.graycomatrix` | upstream-only | pending | pending |
| `feature.graycoprops` | upstream-only | pending | pending |
| `feature.haar_like_feature` | upstream-only | pending | pending |
| `feature.haar_like_feature_coord` | upstream-only | pending | pending |
| `feature.hessian_matrix` | upstream-only | pending | pending |
| `feature.hessian_matrix_det` | upstream-only | pending | pending |
| `feature.hessian_matrix_eigvals` | upstream-only | pending | pending |
| `feature.hog` | upstream-only | pending | pending |
| `feature.learn_gmm` | upstream-only | pending | pending |
| `feature.local_binary_pattern` | upstream-only | pending | pending |
| `feature.match_descriptors` | upstream-only | pending | pending |
| `feature.match_template` | upstream-only | pending | pending |
| `feature.multiblock_lbp` | upstream-only | pending | pending |
| `feature.multiscale_basic_features` | upstream-only | pending | pending |
| `feature.peak_local_max` | upstream-only | pending | pending |
| `feature.plot_matched_features` | upstream-only | pending | pending |
| `feature.shape_index` | upstream-only | pending | pending |
| `feature.structure_tensor` | upstream-only | pending | pending |
| `feature.structure_tensor_eigenvalues` | upstream-only | pending | pending |
| `filters.LPIFilter2D` | upstream-only | pending | pending |
| `filters.apply_hysteresis_threshold` | upstream-only | pending | pending |
| `filters.butterworth` | upstream-only | pending | pending |
| `filters.correlate_sparse` | upstream-only | pending | pending |
| `filters.difference_of_gaussians` | native-partial | yes (partial input contract) | yes |
| `filters.farid` | native-partial | yes (partial input contract) | yes |
| `filters.farid_h` | native-partial | yes (partial input contract) | yes |
| `filters.farid_v` | native-partial | yes (partial input contract) | yes |
| `filters.filter_forward` | upstream-only | pending | pending |
| `filters.filter_inverse` | upstream-only | pending | pending |
| `filters.frangi` | upstream-only | pending | pending |
| `filters.gabor` | upstream-only | pending | pending |
| `filters.gabor_kernel` | upstream-only | pending | pending |
| `filters.gaussian` | native-partial | yes (partial input contract) | yes |
| `filters.hessian` | upstream-only | pending | pending |
| `filters.laplace` | native-partial | yes (partial input contract) | yes |
| `filters.median` | upstream-only | pending | pending |
| `filters.meijering` | upstream-only | pending | pending |
| `filters.prewitt` | native-partial | yes (partial input contract) | yes |
| `filters.prewitt_h` | native-partial | yes (partial input contract) | yes |
| `filters.prewitt_v` | native-partial | yes (partial input contract) | yes |
| `filters.rank_order` | upstream-only | pending | pending |
| `filters.roberts` | native-partial | yes (partial input contract) | yes |
| `filters.roberts_neg_diag` | native-partial | yes (partial input contract) | yes |
| `filters.roberts_pos_diag` | native-partial | yes (partial input contract) | yes |
| `filters.sato` | upstream-only | pending | pending |
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
| `filters.threshold_multiotsu` | upstream-only | pending | pending |
| `filters.threshold_niblack` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_otsu` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_sauvola` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_triangle` | native-partial | yes (partial input contract) | yes |
| `filters.threshold_yen` | native-partial | yes (partial input contract) | yes |
| `filters.try_all_threshold` | upstream-only | pending | pending |
| `filters.unsharp_mask` | native-partial | yes (partial input contract) | yes |
| `filters.wiener` | upstream-only | pending | pending |
| `filters.window` | upstream-only | pending | pending |
| `filters.rank.autolevel` | upstream-only | pending | pending |
| `filters.rank.autolevel_percentile` | upstream-only | pending | pending |
| `filters.rank.enhance_contrast` | upstream-only | pending | pending |
| `filters.rank.enhance_contrast_percentile` | upstream-only | pending | pending |
| `filters.rank.entropy` | upstream-only | pending | pending |
| `filters.rank.equalize` | upstream-only | pending | pending |
| `filters.rank.geometric_mean` | upstream-only | pending | pending |
| `filters.rank.gradient` | upstream-only | pending | pending |
| `filters.rank.gradient_percentile` | upstream-only | pending | pending |
| `filters.rank.majority` | upstream-only | pending | pending |
| `filters.rank.maximum` | upstream-only | pending | pending |
| `filters.rank.mean` | upstream-only | pending | pending |
| `filters.rank.mean_bilateral` | upstream-only | pending | pending |
| `filters.rank.mean_percentile` | upstream-only | pending | pending |
| `filters.rank.median` | upstream-only | pending | pending |
| `filters.rank.minimum` | upstream-only | pending | pending |
| `filters.rank.modal` | upstream-only | pending | pending |
| `filters.rank.noise_filter` | upstream-only | pending | pending |
| `filters.rank.otsu` | upstream-only | pending | pending |
| `filters.rank.percentile` | upstream-only | pending | pending |
| `filters.rank.pop` | upstream-only | pending | pending |
| `filters.rank.pop_bilateral` | upstream-only | pending | pending |
| `filters.rank.pop_percentile` | upstream-only | pending | pending |
| `filters.rank.subtract_mean` | upstream-only | pending | pending |
| `filters.rank.subtract_mean_percentile` | upstream-only | pending | pending |
| `filters.rank.sum` | upstream-only | pending | pending |
| `filters.rank.sum_bilateral` | upstream-only | pending | pending |
| `filters.rank.sum_percentile` | upstream-only | pending | pending |
| `filters.rank.threshold` | upstream-only | pending | pending |
| `filters.rank.threshold_percentile` | upstream-only | pending | pending |
| `filters.rank.windowed_histogram` | upstream-only | pending | pending |
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
| `measure.blur_effect` | upstream-only | pending | pending |
| `measure.centroid` | upstream-only | pending | pending |
| `measure.euler_number` | upstream-only | pending | pending |
| `measure.find_contours` | upstream-only | pending | pending |
| `measure.grid_points_in_poly` | upstream-only | pending | pending |
| `measure.inertia_tensor` | upstream-only | pending | pending |
| `measure.inertia_tensor_eigvals` | upstream-only | pending | pending |
| `measure.intersection_coeff` | upstream-only | pending | pending |
| `measure.label` | upstream-only | pending | pending |
| `measure.manders_coloc_coeff` | upstream-only | pending | pending |
| `measure.manders_overlap_coeff` | upstream-only | pending | pending |
| `measure.marching_cubes` | upstream-only | pending | pending |
| `measure.mesh_surface_area` | upstream-only | pending | pending |
| `measure.moments` | upstream-only | pending | pending |
| `measure.moments_central` | upstream-only | pending | pending |
| `measure.moments_coords` | upstream-only | pending | pending |
| `measure.moments_coords_central` | upstream-only | pending | pending |
| `measure.moments_hu` | upstream-only | pending | pending |
| `measure.moments_normalized` | upstream-only | pending | pending |
| `measure.pearson_corr_coeff` | upstream-only | pending | pending |
| `measure.perimeter` | upstream-only | pending | pending |
| `measure.perimeter_crofton` | upstream-only | pending | pending |
| `measure.points_in_poly` | upstream-only | pending | pending |
| `measure.profile_line` | upstream-only | pending | pending |
| `measure.ransac` | upstream-only | pending | pending |
| `measure.regionprops` | upstream-only | pending | pending |
| `measure.regionprops_table` | upstream-only | pending | pending |
| `measure.shannon_entropy` | upstream-only | pending | pending |
| `measure.subdivide_polygon` | upstream-only | pending | pending |
| `metrics.adapted_rand_error` | upstream-only | pending | pending |
| `metrics.contingency_table` | upstream-only | pending | pending |
| `metrics.hausdorff_distance` | upstream-only | pending | pending |
| `metrics.hausdorff_pair` | upstream-only | pending | pending |
| `metrics.mean_squared_error` | upstream-only | pending | pending |
| `metrics.normalized_mutual_information` | upstream-only | pending | pending |
| `metrics.normalized_root_mse` | upstream-only | pending | pending |
| `metrics.peak_signal_noise_ratio` | upstream-only | pending | pending |
| `metrics.structural_similarity` | upstream-only | pending | pending |
| `metrics.variation_of_information` | upstream-only | pending | pending |
| `morphology.area_closing` | upstream-only | pending | pending |
| `morphology.area_opening` | upstream-only | pending | pending |
| `morphology.ball` | upstream-only | pending | pending |
| `morphology.binary_closing` | native-partial | yes (partial input contract) | yes |
| `morphology.binary_dilation` | native-partial | yes (partial input contract) | yes |
| `morphology.binary_erosion` | native-partial | yes (partial input contract) | yes |
| `morphology.binary_opening` | native-partial | yes (partial input contract) | yes |
| `morphology.black_tophat` | native-partial | yes (partial input contract) | yes |
| `morphology.closing` | native-partial | yes (partial input contract) | yes |
| `morphology.convex_hull_image` | upstream-only | pending | pending |
| `morphology.convex_hull_object` | upstream-only | pending | pending |
| `morphology.diameter_closing` | upstream-only | pending | pending |
| `morphology.diameter_opening` | upstream-only | pending | pending |
| `morphology.diamond` | upstream-only | pending | pending |
| `morphology.dilation` | native-partial | yes (partial input contract) | yes |
| `morphology.disk` | upstream-only | pending | pending |
| `morphology.ellipse` | upstream-only | pending | pending |
| `morphology.erosion` | native-partial | yes (partial input contract) | yes |
| `morphology.flood` | upstream-only | pending | pending |
| `morphology.flood_fill` | upstream-only | pending | pending |
| `morphology.footprint_from_sequence` | upstream-only | pending | pending |
| `morphology.footprint_rectangle` | upstream-only | pending | pending |
| `morphology.h_maxima` | upstream-only | pending | pending |
| `morphology.h_minima` | upstream-only | pending | pending |
| `morphology.isotropic_closing` | upstream-only | pending | pending |
| `morphology.isotropic_dilation` | upstream-only | pending | pending |
| `morphology.isotropic_erosion` | upstream-only | pending | pending |
| `morphology.isotropic_opening` | upstream-only | pending | pending |
| `morphology.label` | upstream-only | pending | pending |
| `morphology.local_maxima` | upstream-only | pending | pending |
| `morphology.local_minima` | upstream-only | pending | pending |
| `morphology.max_tree` | upstream-only | pending | pending |
| `morphology.max_tree_local_maxima` | upstream-only | pending | pending |
| `morphology.medial_axis` | upstream-only | pending | pending |
| `morphology.mirror_footprint` | upstream-only | pending | pending |
| `morphology.octagon` | upstream-only | pending | pending |
| `morphology.octahedron` | upstream-only | pending | pending |
| `morphology.opening` | native-partial | yes (partial input contract) | yes |
| `morphology.pad_footprint` | upstream-only | pending | pending |
| `morphology.reconstruction` | upstream-only | pending | pending |
| `morphology.remove_objects_by_distance` | upstream-only | pending | pending |
| `morphology.remove_small_holes` | upstream-only | pending | pending |
| `morphology.remove_small_objects` | upstream-only | pending | pending |
| `morphology.skeletonize` | upstream-only | pending | pending |
| `morphology.star` | upstream-only | pending | pending |
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
| `restoration.denoise_tv_bregman` | upstream-only | pending | pending |
| `restoration.denoise_tv_chambolle` | upstream-only | pending | pending |
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
| `segmentation.clear_border` | upstream-only | pending | pending |
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
| `segmentation.relabel_sequential` | upstream-only | pending | pending |
| `segmentation.slic` | upstream-only | pending | pending |
| `segmentation.watershed` | upstream-only | pending | pending |
| `transform.AffineTransform` | upstream-only | pending | pending |
| `transform.EssentialMatrixTransform` | upstream-only | pending | pending |
| `transform.EuclideanTransform` | upstream-only | pending | pending |
| `transform.FundamentalMatrixTransform` | upstream-only | pending | pending |
| `transform.PiecewiseAffineTransform` | upstream-only | pending | pending |
| `transform.PolynomialTransform` | upstream-only | pending | pending |
| `transform.ProjectiveTransform` | upstream-only | pending | pending |
| `transform.SimilarityTransform` | upstream-only | pending | pending |
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
| `transform.matrix_transform` | upstream-only | pending | pending |
| `transform.order_angles_golden_ratio` | upstream-only | pending | pending |
| `transform.probabilistic_hough_line` | upstream-only | pending | pending |
| `transform.pyramid_expand` | upstream-only | pending | pending |
| `transform.pyramid_gaussian` | upstream-only | pending | pending |
| `transform.pyramid_laplacian` | upstream-only | pending | pending |
| `transform.pyramid_reduce` | upstream-only | pending | pending |
| `transform.radon` | upstream-only | pending | pending |
| `transform.rescale` | upstream-only | pending | pending |
| `transform.resize` | upstream-only | pending | pending |
| `transform.resize_local_mean` | upstream-only | pending | pending |
| `transform.rotate` | upstream-only | pending | pending |
| `transform.swirl` | upstream-only | pending | pending |
| `transform.warp` | upstream-only | pending | pending |
| `transform.warp_coords` | upstream-only | pending | pending |
| `transform.warp_polar` | upstream-only | pending | pending |
| `util.FailedEstimationAccessError` | upstream-only | pending | pending |
| `util.PendingSkimage2Change` | upstream-only | pending | pending |
| `util.apply_parallel` | upstream-only | pending | pending |
| `util.compare_images` | upstream-only | pending | pending |
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

29 exported class names expose 284 statically discoverable member paths, including inherited methods and properties. Their signatures and defining classes are recorded in `api.json`. These members still require native implementations, numerical/state parity cases and benchmarks; they do not increase native coverage. Constructors are not executed and property getters are not evaluated during inventory. Instance fields and runtime-generated attributes remain unassessed.

| Class | Static member paths | Native member parity |
|---|---:|---|
| `feature.BRIEF` | 2 | pending |
| `feature.CENSURE` | 2 | pending |
| `feature.Cascade` | 11 | pending |
| `feature.ORB` | 4 | pending |
| `feature.SIFT` | 5 | pending |
| `filters.LPIFilter2D` | 2 | pending |
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
| `transform.AffineTransform` | 17 | pending |
| `transform.EssentialMatrixTransform` | 8 | pending |
| `transform.EuclideanTransform` | 15 | pending |
| `transform.FundamentalMatrixTransform` | 8 | pending |
| `transform.PiecewiseAffineTransform` | 7 | pending |
| `transform.PolynomialTransform` | 7 | pending |
| `transform.ProjectiveTransform` | 13 | pending |
| `transform.SimilarityTransform` | 16 | pending |
| `transform.ThinPlateSplineTransform` | 5 | pending |
| `util.FailedEstimationAccessError` | 13 | pending |
| `util.PendingSkimage2Change` | 10 | pending |
