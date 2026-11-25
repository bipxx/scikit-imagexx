// SIFT scale-space and histogram conventions follow scikit-image 0.26.0
// (BSD-3-Clause) and Otero & Delbracio, IPOL 2014, DOI:10.5201/ipol.2014.82.
#include "skimagexx/sift.hpp"

#include "execution.hpp"
#include "resample_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {

constexpr double pi = 3.14159265358979323846264338327950288;
constexpr double tau = 2 * pi;

template<class T>
constexpr PixelType real_type = std::is_same_v<T, float> ? PixelType::float32 : PixelType::float64;

int octaves_for(int height, int width, const SiftOptions& p) {
    if ((p.upsampling != 1 && p.upsampling != 2 && p.upsampling != 4) ||
        p.n_octaves < 1 || p.n_scales < 1 || p.n_scales > 12 ||
        p.n_bins < 6 || p.n_bins > 256 || p.n_hist < 1 || p.n_hist > 16 ||
        p.n_ori < 1 || p.n_ori > 64 || height < 3 || width < 3 ||
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

template<class T>
void blur(const std::vector<T>& input, std::vector<T>& output, int height, int width,
          double sigma) {
    const int radius = int(4 * sigma + 0.5);
    if (radius == 0) {
        output = input;
        return;
    }
    std::vector<double> weights(2 * radius + 1);
    const double exponent = -0.5 / (sigma * sigma);
    for (int x = -radius; x <= radius; ++x)
        weights[x + radius] = std::exp(exponent * double(x * x));
    const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& weight : weights) weight /= sum;
    output.resize(input.size());
    gaussian_nd_into(input.data(), output.data(), real_type<T>, real_type<T>,
                     {std::size_t(height), std::size_t(width)},
                     {{0, weights, GaussianBorder::reflect}, {1, weights, GaussianBorder::reflect}});
}

template<class T>
struct ScaleOctave {
    int height, width;
    std::vector<std::vector<T>> levels;
};

template<class T>
class ScaleSpace {
public:
    ScaleSpace(const void* image, int height, int width, PixelType type, const SiftOptions& p)
        : height_(height * p.upsampling), width_(width * p.upsampling), p_(p),
          sigmas_(p.n_scales + 3) {
        if (!image) throw std::invalid_argument("Non-null SIFT image required");
        std::vector<T> input(std::size_t(height) * width);
        convert_into(image, input.data(), input.size(), type, real_type<T>);
        if ((type == PixelType::float32 || type == PixelType::float64) &&
            std::any_of(input.begin(), input.end(), [](T pixel) { return !std::isfinite(pixel); }))
            throw std::invalid_argument("SIFT requires a finite image");
        if (p.upsampling > 1) {
            std::vector<T> resized(std::size_t(height_) * width_);
            detail::zoom_typed_into(input.data(), resized.data(), real_type<T>,
                                    {std::size_t(height), std::size_t(width)},
                                    {std::size_t(height_), std::size_t(width_)}, {});
            const auto limits = std::minmax_element(input.begin(), input.end());
            for (auto& pixel : resized) pixel = std::clamp(pixel, *limits.first, *limits.second);
            input = std::move(resized);
        }
        const auto initial = p.sigma_in * p.upsampling;
        blur(input, base_, height_, width_, std::sqrt(p.sigma_min * p.sigma_min - initial * initial));
        for (int s = 0; s < p.n_scales + 3; ++s)
            sigmas_[s] = std::pow(2.0, double(s) / p.n_scales) * p.sigma_min;
    }

    ScaleOctave<T> next(bool more) {
        ScaleOctave<T> out{height_, width_, std::vector<std::vector<T>>(sigmas_.size())};
        out.levels[0] = std::move(base_);
        for (std::size_t s = 1; s < sigmas_.size(); ++s) {
            const auto before = sigmas_[s - 1], after = sigmas_[s];
            blur(out.levels[s - 1], out.levels[s], height_, width_,
                 std::sqrt(after * after - before * before));
        }
        if (more) {
            const int height = (height_ + 1) / 2, width = (width_ + 1) / 2;
            base_.resize(std::size_t(height) * width);
            const auto& level = out.levels[p_.n_scales];
            for (int r = 0; r < height; ++r)
                for (int c = 0; c < width; ++c)
                    base_[std::size_t(r) * width + c] = level[std::size_t(2 * r) * width_ + 2 * c];
            height_ = height;
            width_ = width;
        }
        return out;
    }

    double sigma(int scale, T offset, double delta) const {
        return sigmas_[scale] * delta * std::pow(sigmas_[1] / sigmas_[0], offset);
    }

private:
    int height_, width_;
    const SiftOptions& p_;
    std::vector<T> base_;
    std::vector<double> sigmas_;
};

template<class T>
struct Dog {
    int height, width, scales;
    std::vector<T> values;

    explicit Dog(const ScaleOctave<T>& octave)
        : height(octave.height), width(octave.width), scales(int(octave.levels.size()) - 1),
          values(std::size_t(height) * width * scales) {
        detail::partition(std::size_t(height) * width, detail::active_workers(values.size()),
            [&](auto first, auto last, auto) {
                for (auto k = first; k < last; ++k)
                    for (int s = 0; s < scales; ++s)
                        values[k * scales + s] = octave.levels[s + 1][k] - octave.levels[s][k];
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
        for (int c = 1; c < limits[1]; ++c) {
            for (int s = 1; s < limits[2]; ++s) {
                const auto* center = dog.at(r, c, s);
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
std::pair<T,T> gradient(const std::vector<T>& level, int height, int width, int r, int c) {
    const auto index = std::size_t(r) * width + c;
    const auto* at = level.data() + index;
    const T dr = r == 0 ? at[width] - at[0] : r == height - 1 ? at[0] - at[-width] :
                 T(0.5) * (at[width] - at[-width]);
    const T dc = c == 0 ? at[1] - at[0] : c == width - 1 ? at[0] - at[-1] :
                 T(0.5) * (at[1] - at[-1]);
    return {dr, dc};
}

template<class T>
void orientations(const ScaleOctave<T>& octave, const SiftOptions& p, double delta,
                  SiftKeypoint& key, std::vector<SiftKeypoint>& extra) {
    const double y = key.row / delta, x = key.col / delta, sigma = key.sigma / delta;
    const double width = p.lambda_ori * sigma;
    const auto box = patch(y, x, 3 * width, octave.height, octave.width);
    std::vector<T> hist(p.n_bins, T(0)), smooth(p.n_bins);
    for (int r = box.r0; r <= box.r1; ++r) {
        const T dy = T(r - y);
        for (int c = box.c0; c <= box.c1; ++c) {
            const T dx = T(c - x);
            const auto [dr, dc] = gradient(octave.levels[key.scale], octave.height, octave.width, r, c);
            T theta = std::atan2(dc, dr);
            if (!std::isfinite(theta)) continue;
            if (theta < T(0)) theta += T(tau);
            const int bin = int(std::floor(theta / T(tau) * T(p.n_bins) + T(0.5))) % p.n_bins;
            const double weight = std::exp(double(dy * dy + dx * dx) / (-2 * width * width));
            hist[bin] = T(hist[bin] + weight * std::sqrt(dr * dr + dc * dc));
        }
    }
    const T third = T(1.0 / 3);
    for (int pass = 0; pass < 6; ++pass) {
        for (int j = 0; j < p.n_bins; ++j)
            smooth[j] = third * hist[(j + p.n_bins - 1) % p.n_bins] +
                        third * hist[j] + third * hist[(j + 1) % p.n_bins];
        std::swap(hist, smooth);
    }
    const T threshold = T(p.c_max) * *std::max_element(hist.begin(), hist.end());
    bool primary = true;
    for (int j = 0; j < p.n_bins; ++j) {
        const T left = hist[(j + p.n_bins - 1) % p.n_bins], mid = hist[j];
        const T right = hist[(j + 1) % p.n_bins];
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
                const SiftKeypoint& key, T* histogram, std::uint8_t* output) {
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
    std::array<T, 16> spatial{};
    std::array<T, 64> angles{};
    for (int j = 0; j < p.n_hist; ++j)
        spatial[j] = T(j + 1 - (1 + p.n_hist) / 2.0) * spacing;
    for (int j = 0; j < p.n_ori; ++j) angles[j] = (T(tau) * T(j + 1)) / T(p.n_ori);
    for (int r = box.r0; r < box.r1; ++r) {
        const T dy = T(r) - T(y);
        for (int c = box.c0; c < box.c1; ++c) {
            const T dx = T(c) - T(x);
            const T yr = cos * dy + sin * dx, xr = -sin * dy + cos * dx;
            if (std::max(std::abs(yr), std::abs(xr)) >= T(radius)) continue;
            const auto [dr, dc] = gradient(octave.levels[key.scale], octave.height, octave.width, r, c);
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
            const int next = (bin + 1) % p.n_ori;
            for (int i = 0; i < p.n_hist; ++i) {
                const T v = std::abs(spatial[i] - yr);
                if (v > spacing) continue;
                for (int j = 0; j < p.n_hist; ++j) {
                    const T u = std::abs(spatial[j] - xr);
                    if (u > spacing) continue;
                    const T weight = (1 - inverse * v) * (1 - inverse * u) * magnitude;
                    const T portion = weight * angular * nearest;
                    T* cell = histogram + (i * p.n_hist + j) * p.n_ori;
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
std::vector<std::uint8_t> describe_keys(const ScaleOctave<T>& octave, const SiftOptions& p,
                                       double delta, const std::vector<SiftKeypoint>& keys,
                                       const std::vector<std::size_t>& indices, bool& zero_norm) {
    const int columns = p.n_hist * p.n_hist * p.n_ori;
    std::vector<std::uint8_t> output(indices.size() * columns);
    if (indices.empty()) return output;
    const int workers = indices.size() >= 32 ? get_num_threads() : 1;
    std::vector<T> working(std::size_t(workers) * columns);
    std::vector<unsigned char> degenerate(workers, 0);
    detail::partition(indices.size(), workers, [&](auto first, auto last, auto worker) {
        for (auto i = first; i < last; ++i)
            if (!descriptor(octave, p, delta, keys[indices[i]], working.data() + worker * columns,
                            output.data() + i * columns))
                degenerate[worker] = 1;
    });
    zero_norm = zero_norm || std::find(degenerate.begin(), degenerate.end(), 1) != degenerate.end();
    return output;
}

template<class T>
SiftResult detect(const void* image, int height, int width, PixelType type,
                  const SiftOptions& p, int octaves, bool describe) {
    SiftResult result;
    std::vector<SiftKeypoint> extra;
    std::vector<std::uint8_t> extra_descriptors;
    ScaleSpace<T> space(image, height, width, type, p);
    for (int o = 0; o < octaves; ++o) {
        auto octave = space.next(o + 1 < octaves);
        const auto first = result.keypoints.size();
        const auto extra_first = extra.size();
        localize(Dog<T>(octave), space, p, height, width, o, result.keypoints, result.promoted_positions);
        const double delta = std::ldexp(1.0 / p.upsampling, o);
        for (auto j = first; j < result.keypoints.size(); ++j)
            orientations(octave, p, delta, result.keypoints[j], extra);
        if (describe) {
            std::vector<std::size_t> indices(result.keypoints.size() - first);
            std::iota(indices.begin(), indices.end(), first);
            auto rows = describe_keys(octave, p, delta, result.keypoints, indices,
                                      result.descriptor_has_zero_norm);
            result.descriptors.insert(result.descriptors.end(), rows.begin(), rows.end());
            indices.resize(extra.size() - extra_first);
            std::iota(indices.begin(), indices.end(), extra_first);
            rows = describe_keys(octave, p, delta, extra, indices, result.descriptor_has_zero_norm);
            extra_descriptors.insert(extra_descriptors.end(), rows.begin(), rows.end());
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
    const int columns = p.n_hist * p.n_hist * p.n_ori;
    std::vector<std::uint8_t> output(keys.size() * columns);
    ScaleSpace<T> space(image, height, width, type, p);
    for (int o = 0; o < octaves; ++o) {
        auto octave = space.next(o + 1 < octaves);
        const double delta = std::ldexp(1.0 / p.upsampling, o);
        std::vector<std::size_t> indices;
        for (std::size_t j = 0; j < keys.size(); ++j)
            if (keys[j].octave == o) indices.push_back(j);
        auto rows = describe_keys(octave, p, delta, keys, indices, zero_norm);
        for (std::size_t j = 0; j < indices.size(); ++j)
            std::copy_n(rows.data() + j * columns, columns, output.data() + indices[j] * columns);
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
