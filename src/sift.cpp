// SIFT scale-space and histogram conventions follow scikit-image 0.26.0
// (BSD-3-Clause) and Otero & Delbracio, IPOL 2014, DOI:10.5201/ipol.2014.82.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif
#include "skimagexx/sift.hpp"

#include "execution.hpp"
#include "resample_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {

constexpr double pi = 3.14159265358979323846264338327950288;
constexpr double tau = 2 * pi;
constexpr int max_scales = 12, max_bins = 256, max_spatial_bins = 16, max_angular_bins = 64;

template<class T>
constexpr PixelType real_type = std::is_same_v<T, float> ? PixelType::float32 : PixelType::float64;

int octaves_for(int height, int width, const SiftOptions& p) {
    if ((p.upsampling != 1 && p.upsampling != 2 && p.upsampling != 4) ||
        p.n_octaves < 1 || p.n_scales < 1 || p.n_scales > max_scales ||
        p.n_bins < 6 || p.n_bins > max_bins || p.n_hist < 1 || p.n_hist > max_spatial_bins ||
        p.n_ori < 1 || p.n_ori > max_angular_bins || height < 3 || width < 3 ||
        height > 16384 || width > 16384 ||
        !(p.sigma_min >= p.sigma_in * p.upsampling && p.sigma_min > 0 &&
          p.sigma_min <= 64 && p.sigma_in >= 0 && p.c_dog > 0 &&
          std::isfinite(p.c_dog) && p.c_edge > 0 && std::isfinite(p.c_edge) &&
          p.lambda_ori > 0 && p.lambda_ori <= 64 && p.lambda_descr > 0 &&
          p.lambda_descr <= 64 && p.c_max > 0 && p.c_max <= 1) ||
        std::size_t(height) * width * p.upsampling * p.upsampling >
            (std::size_t{128} << 20) / (p.n_scales + 3))
        throw std::invalid_argument("Unsupported SIFT options or image dimensions");
    const int smallest = std::min(height, width) * p.upsampling;
    if (smallest < 12) throw std::invalid_argument("SIFT scale space requires at least 12 pixels");
    return std::min(p.n_octaves, int(std::log2(smallest / 12.0) + 1));
}

struct BlurPasses {
    std::vector<GaussianPass> vertical, horizontal;

    explicit BlurPasses(double sigma) {
        const int radius = int(4 * sigma + 0.5);
        if (radius == 0) return;
        std::vector<double> weights(2 * radius + 1);
        const double exponent = -0.5 / (sigma * sigma);
        for (int x = -radius; x <= radius; ++x)
            weights[x + radius] = std::exp(exponent * double(x * x));
        const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (auto& weight : weights) weight /= sum;
        vertical.push_back({0, weights, GaussianBorder::reflect});
        horizontal.push_back({1, std::move(weights), GaussianBorder::reflect});
    }
};

template<class T>
struct ScaleOctave {
    int height, width;
    std::vector<std::vector<T>> levels;
};

template<class T>
class ScaleSpace {
public:
    ScaleSpace(const void* image, int height, int width, PixelType type, const SiftOptions& p)
        : p_(p), octave_{height * p.upsampling, width * p.upsampling,
                        std::vector<std::vector<T>>(p.n_scales + 3)},
          shape_{std::size_t(octave_.height), std::size_t(octave_.width)},
          sigmas_(p.n_scales + 3) {
        if (!image) throw std::invalid_argument("Non-null SIFT image required");
        std::vector<T> input(std::size_t(height) * width);
        convert_into(image, input.data(), input.size(), type, real_type<T>);
        if ((type == PixelType::float32 || type == PixelType::float64) &&
            std::any_of(input.begin(), input.end(), [](T pixel) { return !std::isfinite(pixel); }))
            throw std::invalid_argument("SIFT requires a finite image");
        if (p.upsampling > 1) {
            std::vector<T> resized(shape_[0] * shape_[1]);
            detail::zoom_typed_into(input.data(), resized.data(), real_type<T>,
                                    {std::size_t(height), std::size_t(width)},
                                    shape_, {});
            const auto limits = std::minmax_element(input.begin(), input.end());
            for (auto& pixel : resized) pixel = std::clamp(pixel, *limits.first, *limits.second);
            input = std::move(resized);
        }
        const auto initial = p.sigma_in * p.upsampling;
        blur(input, octave_.levels[0], BlurPasses(std::sqrt(p.sigma_min * p.sigma_min - initial * initial)));
        for (int s = 0; s < p.n_scales + 3; ++s)
            sigmas_[s] = std::pow(2.0, double(s) / p.n_scales) * p.sigma_min;
        passes_.reserve(sigmas_.size() - 1);
        for (std::size_t s = 1; s < sigmas_.size(); ++s) {
            const auto before = sigmas_[s - 1], after = sigmas_[s];
            passes_.emplace_back(std::sqrt(after * after - before * before));
        }
    }

    const ScaleOctave<T>& next(bool downsample) {
        auto& [height, width, levels] = octave_;
        if (downsample) {
            const int previous_width = width;
            height = (height + 1) / 2;
            width = (width + 1) / 2;
            shape_[0] = height;
            shape_[1] = width;
            levels[0].resize(std::size_t(height) * width);
            for (int r = 0; r < height; ++r) {
                const T* source = levels[p_.n_scales].data() + std::size_t(2 * r) * previous_width;
                T* destination = levels[0].data() + std::size_t(r) * width;
                for (int c = 0; c < width; ++c)
                    destination[c] = source[2 * c];
            }
        }
        for (std::size_t s = 1; s < levels.size(); ++s)
            blur(levels[s - 1], levels[s], passes_[s - 1]);
        return octave_;
    }

    double sigma(int scale, T offset, double delta) const {
        return sigmas_[scale] * delta * std::pow(sigmas_[1] / sigmas_[0], offset);
    }

private:
    void blur(const std::vector<T>& input, std::vector<T>& output, const BlurPasses& passes) {
        if (passes.vertical.empty()) {
            output = input;
            return;
        }
        output.resize(input.size());
        if (!temporary_) temporary_.reset(new T[input.size()]);
        // Keep the shared Gaussian kernel and the rounding after each axis.
        gaussian_nd_into(input.data(), temporary_.get(), real_type<T>, real_type<T>, shape_, passes.vertical);
        gaussian_nd_into(temporary_.get(), output.data(), real_type<T>, real_type<T>, shape_, passes.horizontal);
    }

    const SiftOptions& p_;
    ScaleOctave<T> octave_;
    std::vector<std::size_t> shape_;
    std::vector<double> sigmas_;
    std::vector<BlurPasses> passes_;
    std::unique_ptr<T[]> temporary_;
};

template<class T>
struct Dog {
    int height, width, scales;
    std::vector<T> values;

    void update(const ScaleOctave<T>& octave) {
        height = octave.height;
        width = octave.width;
        scales = int(octave.levels.size()) - 1;
        values.resize(std::size_t(height) * width * scales);
        std::array<const T*, max_scales + 3> levels;
        for (int s = 0; s <= scales; ++s)
            levels[s] = octave.levels[s].data();
        detail::partition(std::size_t(height) * width, detail::active_workers(values.size()),
            [&](auto first, auto last, auto) {
                auto* destination = values.data() + first * scales;
                for (auto k = first; k < last; ++k, destination += scales) {
                    T lower = levels[0][k];
                    for (int s = 0; s < scales; ++s) {
                        const T upper = levels[s + 1][k];
                        destination[s] = upper - lower;
                        lower = upper;
                    }
                }
            });
    }

    const T* at(int r, int c, int s) const {
        return values.data() + (std::size_t(r) * width + c) * scales + s;
    }
};

template<class T>
struct Refinement {
    std::array<T, 3> off, gradient;
    T h00, h11, h01;
};

template<class T>
Refinement<T> refine(const T* p, std::ptrdiff_t row, std::ptrdiff_t col) {
    const T g0 = (p[row] - p[-row]) * T(0.5);
    const T g1 = (p[col] - p[-col]) * T(0.5);
    const T g2 = (p[1] - p[-1]) * T(0.5);
    const T center = 2 * p[0];
    const T h00 = p[-row] + p[row] - center;
    const T h11 = p[-col] + p[col] - center;
    const T h22 = p[-1] + p[1] - center;
    const T h01 = T(0.25) * (p[row + col] - p[-row + col] - p[row - col] + p[-row - col]);
    const T h02 = T(0.25) * (p[row + 1] - p[row - 1] + p[-row - 1] - p[-row + 1]);
    const T h12 = T(0.25) * (p[col + 1] - p[col - 1] + p[-col - 1] - p[-col + 1]);
    T det = h00 * h11 * h22;
    det -= h00 * h12 * h12;
    det -= h01 * h01 * h22;
    det += 2 * h01 * h02 * h12;
    det -= h02 * h02 * h11;
    if (det == T(0)) {
        const T infinity = std::numeric_limits<T>::infinity();
        return {{infinity, infinity, infinity}, {g0, g1, g2}, h00, h11, h01};
    }
    const T aa = (h11 * h22 - h12 * h12) / det, ab = (h02 * h12 - h01 * h22) / det;
    const T ac = (h01 * h12 - h02 * h11) / det, bb = (h00 * h22 - h02 * h02) / det;
    const T bc = (h01 * h02 - h00 * h12) / det, cc = (h00 * h11 - h01 * h01) / det;
    return {{-aa * g0 - ab * g1 - ac * g2, -ab * g0 - bb * g1 - bc * g2,
             -ac * g0 - bc * g1 - cc * g2}, {g0, g1, g2}, h00, h11, h01};
}

template<class T>
void localize(const Dog<T>& dog, const ScaleSpace<T>& space, const SiftOptions& p,
              int height, int width, int octave, std::vector<SiftKeypoint>& keys, bool& promoted) {
    const double delta = std::ldexp(1.0 / p.upsampling, octave);
    const double contrast = (std::pow(2.0, 1.0 / p.n_scales) - 1) /
                            (std::pow(2.0, 1.0 / 3) - 1) * p.c_dog;
    const double edge = (p.c_edge + 1) * (p.c_edge + 1) / p.c_edge;
    const auto col = std::ptrdiff_t(dog.scales), row = col * dog.width;
    std::array<std::ptrdiff_t, 26> neighbors{};
    std::size_t n = 0;
    for (int r = -1; r <= 1; ++r)
        for (int c = -1; c <= 1; ++c)
            for (int s = -1; s <= 1; ++s)
                if (r || c || s) neighbors[n++] = r * row + c * col + s;
    bool any = false;
    const std::array<int, 3> limits{dog.height - 1, dog.width - 1, dog.scales - 1};
    for (int r = 1; r < limits[0]; ++r) {
        const T* pixel = dog.at(r, 1, 0);
        for (int c = 1; c < limits[1]; ++c, pixel += col) {
            for (int s = 1; s < limits[2]; ++s) {
                const auto* center = pixel + s;
                if (std::abs(center[0]) < contrast * 0.8) continue;
                if (center[0] == center[neighbors[0]]) continue;
                const bool minimum = center[0] < center[neighbors[0]];
                const auto fails = [&](auto offset) {
                    return minimum ? center[offset] <= center[0] : center[offset] >= center[0];
                };
                if (std::any_of(neighbors.begin() + 1, neighbors.end(), fails)) continue;
                any = true;
                std::array<int, 3> key{r, c, s};
                Refinement<T> fit{};
                for (int attempt = 0; attempt < 5; ++attempt) {
                    center = dog.at(key[0], key[1], key[2]);
                    fit = refine(center, row, col);
                    if (attempt == 4) break;
                    const auto previous = key;
                    for (int axis = 0; axis < 3; ++axis) {
                        if (fit.off[axis] > T(0.6) && key[axis] + 1 < limits[axis]) ++key[axis];
                        if (fit.off[axis] < T(-0.6) && key[axis] - 1 > 0) --key[axis];
                    }
                    if (key == previous) break;
                }
                if (!std::all_of(fit.off.begin(), fit.off.end(),
                                 [](T x) { return std::abs(x) < T(0.6); })) continue;
                T value = center[0];
                for (int axis = 0; axis < 3; ++axis) value += T(0.5) * fit.gradient[axis] * fit.off[axis];
                if (!(std::abs(value) > T(contrast))) continue;
                const T trace = fit.h00 + fit.h11;
                const T determinant = fit.h00 * fit.h11 - fit.h01 * fit.h01;
                if (determinant == T(0) || !(std::abs(trace * trace / determinant) <= edge)) continue;
                const double y = T((key[0] + double(fit.off[0])) * delta);
                const double x = T((key[1] + double(fit.off[1])) * delta);
                const double sigma = space.sigma(key[2], fit.off[2], delta);
                if (y - sigma > 0 && x - sigma > 0 && y + sigma < height && x + sigma < width)
                    keys.push_back({y, x, sigma, 0, key[2], octave});
            }
        }
    }
    promoted = promoted || !any;
}

struct Patch {
    int r0, c0, r1, c1;
};

Patch patch(double y, double x, double radius, int height, int width) {
    return {int(std::clamp(y - radius + 0.5, 0.0, double(height - 1))),
            int(std::clamp(x - radius + 0.5, 0.0, double(width - 1))),
            int(std::clamp(y + radius + 0.5, 0.0, double(height - 1))),
            int(std::clamp(x + radius + 0.5, 0.0, double(width - 1)))};
}

template<class T>
struct GradientRow {
    const T *previous, *current, *next;
    T row_scale;
    int last_column;

    GradientRow(const std::vector<T>& level, int height, int width, int r)
        : previous(level.data() + std::size_t(std::max(0, r - 1)) * width),
          current(level.data() + std::size_t(r) * width),
          next(level.data() + std::size_t(std::min(height - 1, r + 1)) * width),
          row_scale(r == 0 || r == height - 1 ? T(1) : T(0.5)), last_column(width - 1) {}

    std::pair<T, T> at(int c) const {
        const T dr = row_scale * (next[c] - previous[c]);
        const T dc = c == 0 ? current[1] - current[0] :
                     c == last_column ? current[c] - current[c - 1] :
                     T(0.5) * (current[c + 1] - current[c - 1]);
        return {dr, dc};
    }
};

template<class T>
void orientations(const ScaleOctave<T>& octave, const SiftOptions& p, double delta,
                  SiftKeypoint& key, std::vector<SiftKeypoint>& extra,
                  T* hist, T* smooth) {
    const double y = key.row / delta, x = key.col / delta, sigma = key.sigma / delta;
    const double width = p.lambda_ori * sigma;
    const auto box = patch(y, x, 3 * width, octave.height, octave.width);
    std::fill_n(hist, p.n_bins, T(0));
    const auto& level = octave.levels[key.scale];
    const double gaussian = -2 * width * width;
    for (int r = box.r0; r <= box.r1; ++r) {
        const T dy = T(r - y);
        const T dy_squared = dy * dy;
        const GradientRow<T> gradients(level, octave.height, octave.width, r);
        for (int c = box.c0; c <= box.c1; ++c) {
            const T dx = T(c - x);
            const auto [dr, dc] = gradients.at(c);
            T theta = std::atan2(dc, dr);
            if (!std::isfinite(theta)) continue;
            if (theta < T(0)) theta += T(tau);
            const int bin = int(std::floor(theta / T(tau) * T(p.n_bins) + T(0.5))) % p.n_bins;
            const double weight = std::exp(double(dy_squared + dx * dx) / gaussian);
            hist[bin] = T(hist[bin] + weight * std::sqrt(dr * dr + dc * dc));
        }
    }
    const T third = T(1.0 / 3);
    const int last = p.n_bins - 1;
    for (int pass = 0; pass < 6; ++pass) {
        smooth[0] = third * hist[last] + third * hist[0] + third * hist[1];
        for (int j = 1; j < last; ++j)
            smooth[j] = third * hist[j - 1] + third * hist[j] + third * hist[j + 1];
        smooth[last] = third * hist[last - 1] + third * hist[last] + third * hist[0];
        std::swap(hist, smooth);
    }
    const T threshold = T(p.c_max) * *std::max_element(hist, hist + p.n_bins);
    bool primary = true;
    for (int j = 0; j < p.n_bins; ++j) {
        const T left = hist[j == 0 ? last : j - 1], mid = hist[j];
        const T right = hist[j == last ? 0 : j + 1];
        if (mid < threshold || mid < left || mid < right) continue;
        const T denominator = 2 * (left + right - 2 * mid);
        const T offset = denominator == T(0) ? std::numeric_limits<T>::quiet_NaN() :
                                             (left - right) / denominator;
        double angle = (j + double(offset) + 0.5) * tau / p.n_bins;
        if (angle > pi) angle -= tau;
        if (primary) {
            key.orientation = T(angle);
            primary = false;
        } else {
            extra.push_back(key);
            extra.back().orientation = angle;
        }
    }
}

template<class T>
bool descriptor(const ScaleOctave<T>& octave, const SiftOptions& p, double delta,
                const SiftKeypoint& key, const T* angles, T* histogram, std::uint8_t* output) {
    const int count = p.n_hist * p.n_hist * p.n_ori;
    std::fill(histogram, histogram + count, T(0));
    if (!std::isfinite(key.orientation)) {
        std::fill(output, output + count, std::uint8_t(0));
        return false;
    }
    const double y = key.row / delta, x = key.col / delta, sigma = key.sigma / delta;
    const double radius = p.lambda_descr * (1 + 1.0 / p.n_hist) * sigma;
    const auto box = patch(y, x, std::sqrt(2.0) * radius, octave.height, octave.width);
    const double lam_sig = p.lambda_descr * sigma;
    const T spacing = T(2 * lam_sig / p.n_hist), inverse = T(1) / spacing;
    const T cos = T(std::cos(key.orientation)), sin = T(std::sin(key.orientation));
    const T gaussian = T(-2 * lam_sig * lam_sig), angular = T(p.n_ori / tau);
    std::array<T, max_spatial_bins> spatial;
    for (int j = 0; j < p.n_hist; ++j)
        spatial[j] = T(j + 1 - (1 + p.n_hist) / 2.0) * spacing;
    const auto& level = octave.levels[key.scale];
    const int row_stride = p.n_hist * p.n_ori;
    for (int r = box.r0; r < box.r1; ++r) {
        const T dy = T(r) - T(y);
        const T rotated_y = cos * dy, rotated_x = -sin * dy;
        const GradientRow<T> gradients(level, octave.height, octave.width, r);
        for (int c = box.c0; c < box.c1; ++c) {
            const T dx = T(c) - T(x);
            const T yr = rotated_y + sin * dx, xr = rotated_x + cos * dx;
            if (std::max(std::abs(yr), std::abs(xr)) >= T(radius)) continue;
            const auto [dr, dc] = gradients.at(c);
            const T theta = std::atan2(dc, dr) - T(key.orientation);
            const T magnitude = std::sqrt(dr * dr + dc * dc) * std::exp((yr * yr + xr * xr) / gaussian);
            int bin = 0;
            T nearest = T(100);
            for (int j = 0; j < p.n_ori; ++j) {
                T distance = angles[j] - theta;
                if (distance > T(tau)) distance -= T(tau);
                else if (distance < T(0)) distance += T(tau);
                if (distance < nearest) {
                    bin = j;
                    nearest = distance;
                }
            }
            const int next = bin + 1 == p.n_ori ? 0 : bin + 1;
            std::array<int, max_spatial_bins> offsets;
            std::array<T, max_spatial_bins> horizontal;
            int used = 0;
            for (int j = 0; j < p.n_hist; ++j) {
                const T u = std::abs(spatial[j] - xr);
                if (u > spacing) continue;
                offsets[used] = j * p.n_ori;
                horizontal[used++] = 1 - inverse * u;
            }
            for (int i = 0; i < p.n_hist; ++i) {
                const T v = std::abs(spatial[i] - yr);
                if (v > spacing) continue;
                const T vertical = 1 - inverse * v;
                T* row = histogram + i * row_stride;
                for (int j = 0; j < used; ++j) {
                    const T weight = vertical * horizontal[j] * magnitude;
                    const T portion = weight * angular * nearest;
                    T* cell = row + offsets[j];
                    cell[bin] += portion;
                    cell[next] += weight - portion;
                }
            }
        }
    }
    const auto norm = [&] {
        return std::sqrt(std::inner_product(histogram, histogram + count, histogram, T(0)));
    };
    const T limit = T(0.2) * norm();
    for (int j = 0; j < count; ++j) histogram[j] = std::min(histogram[j], limit);
    const T length = norm();
    if (!(length > T(0) && std::isfinite(length))) {
        std::fill(output, output + count, std::uint8_t(0));
        return false;
    }
    for (int j = 0; j < count; ++j) {
        const T value = 512 * histogram[j] / length;
        output[j] = value > T(0) ? std::uint8_t(std::min(value, T(255))) : std::uint8_t(0);
    }
    return true;
}

template<class T>
class Descriptors {
public:
    explicit Descriptors(const SiftOptions& p) : columns(p.n_hist * p.n_hist * p.n_ori) {
        for (int j = 0; j < p.n_ori; ++j) angles_[j] = (T(tau) * T(j + 1)) / T(p.n_ori);
    }

    void write(const ScaleOctave<T>& octave, const SiftOptions& p, double delta,
               const SiftKeypoint* keys, std::size_t count, std::uint8_t* output,
               bool& zero_norm, const std::size_t* indices = nullptr) {
        const int workers = count >= 32 ? get_num_threads() : 1;
        if (degenerate_.size() < std::size_t(workers)) {
            working_.resize(std::size_t(workers) * columns);
            degenerate_.resize(workers);
        }
        std::fill_n(degenerate_.begin(), workers, std::uint8_t{0});
        detail::partition(count, workers, [&](auto first, auto last, auto worker) {
            T* histogram = working_.data() + worker * columns;
            bool degenerate = false;
            for (auto i = first; i < last; ++i) {
                const auto index = indices ? indices[i] : i;
                if (!descriptor(octave, p, delta, keys[index], angles_.data(), histogram,
                                output + index * columns))
                    degenerate = true;
            }
            degenerate_[worker] = degenerate;
        });
        const auto end = degenerate_.begin() + workers;
        zero_norm = zero_norm || std::find(degenerate_.begin(), end, 1) != end;
    }

    const int columns;

private:
    std::array<T, max_angular_bins> angles_;
    std::vector<T> working_;
    std::vector<std::uint8_t> degenerate_;
};

template<class T>
SiftResult detect(const void* image, int height, int width, PixelType type,
                  const SiftOptions& p, int octaves, bool describe) {
    SiftResult result;
    std::vector<SiftKeypoint> extra;
    std::vector<std::uint8_t> extra_descriptors;
    std::array<T, max_bins> hist, smooth;
    ScaleSpace<T> space(image, height, width, type, p);
    Dog<T> dog;
    Descriptors<T> descriptors(p);
    const auto append_descriptors = [&](const auto& keys, std::size_t first, auto& output,
                                        const auto& octave, double delta) {
        const std::size_t count = keys.size() - first;
        if (!count) return;
        const std::size_t offset = output.size();
        output.resize(offset + count * descriptors.columns);
        descriptors.write(octave, p, delta, keys.data() + first, count, output.data() + offset,
                          result.descriptor_has_zero_norm);
    };
    for (int o = 0; o < octaves; ++o) {
        const auto& octave = space.next(o > 0);
        const auto first = result.keypoints.size();
        const auto extra_first = extra.size();
        dog.update(octave);
        localize(dog, space, p, height, width, o, result.keypoints, result.promoted_positions);
        const double delta = std::ldexp(1.0 / p.upsampling, o);
        for (auto j = first; j < result.keypoints.size(); ++j)
            orientations(octave, p, delta, result.keypoints[j], extra, hist.data(), smooth.data());
        if (describe) {
            append_descriptors(result.keypoints, first, result.descriptors, octave, delta);
            append_descriptors(extra, extra_first, extra_descriptors, octave, delta);
        }
    }
    result.multiple_orientations = !extra.empty();
    result.keypoints.insert(result.keypoints.end(), extra.begin(), extra.end());
    result.descriptors.insert(result.descriptors.end(), extra_descriptors.begin(), extra_descriptors.end());
    return result;
}

template<class T>
std::vector<std::uint8_t> extract(const void* image, int height, int width, PixelType type,
                                  const SiftOptions& p, int octaves,
                                  const std::vector<SiftKeypoint>& keys, bool& zero_norm) {
    Descriptors<T> descriptors(p);
    std::vector<std::uint8_t> output(keys.size() * descriptors.columns);
    std::vector<std::size_t> ends(octaves + 1, 0);
    for (const auto& key : keys) ++ends[key.octave + 1];
    std::partial_sum(ends.begin(), ends.end(), ends.begin());
    std::vector<std::size_t> indices(keys.size());
    for (std::size_t j = 0; j < keys.size(); ++j)
        indices[ends[keys[j].octave]++] = j;

    ScaleSpace<T> space(image, height, width, type, p);
    for (int o = 0; o < octaves; ++o) {
        const auto first = o == 0 ? 0 : ends[o - 1];
        const auto count = ends[o] - first;
        const auto& octave = space.next(o > 0);
        const double delta = std::ldexp(1.0 / p.upsampling, o);
        if (count)
            descriptors.write(octave, p, delta, keys.data(), count, output.data(), zero_norm,
                              indices.data() + first);
    }
    return output;
}

} // namespace

SiftResult sift_detect(const void* image, int height, int width, PixelType type,
                       const SiftOptions& options, bool describe) {
    const int octaves = octaves_for(height, width, options);
    if (type == PixelType::float32) return detect<float>(image, height, width, type, options, octaves, describe);
    return detect<double>(image, height, width, type, options, octaves, describe);
}

std::vector<std::uint8_t> sift_extract(const void* image, int height, int width,
                                      PixelType type, const SiftOptions& options,
                                      const std::vector<SiftKeypoint>& keys, bool* descriptor_has_zero_norm) {
    const int octaves = octaves_for(height, width, options);
    for (const auto& key : keys)
        if (key.scale < 0 || key.scale >= options.n_scales + 3 ||
            key.octave < 0 || key.octave >= octaves ||
            !(std::abs(key.row) <= 16384 && std::abs(key.col) <= 16384) ||
            !(key.sigma > 0 && key.sigma <= 16384))
            throw std::invalid_argument("Invalid SIFT keypoint");
    bool zero_norm = false;
    auto output = type == PixelType::float32
        ? extract<float>(image, height, width, type, options, octaves, keys, zero_norm)
        : extract<double>(image, height, width, type, options, octaves, keys, zero_norm);
    if (descriptor_has_zero_norm) *descriptor_has_zero_norm = zero_norm;
    return output;
}

} // namespace skimagexx
