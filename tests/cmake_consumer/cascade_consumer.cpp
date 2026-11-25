#include "skimagexx/cascade.hpp"

#include <array>
#include <vector>

int main() {
    constexpr auto xml = R"(<opencv_storage><cascade type_id="opencv-cascade-classifier">
    <featureType>LBP</featureType><height>3</height><width>3</width><stageNum>1</stageNum>
    <stages><_><stageThreshold>0</stageThreshold><maxWeakCount>1</maxWeakCount>
    <weakClassifiers><_><internalNodes>0 -1 0 0 0 0 0 0 0 0 -2147483648</internalNodes>
    <leafValues>1 -1</leafValues></_></weakClassifiers></_></stages>
    <features><_><rect>0 0 1 1</rect></_></features></cascade></opencv_storage>)";
    skimagexx::Cascade model(xml);
    if (model.stages_number != 1 || model.stumps_number != 1 || model.features_number != 1 ||
        model.window_width != 3 || model.window_height != 3) return 1;
    constexpr std::array<float, 16> integral{1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16};
    const std::vector<float> scales{1.f};
    const auto boxes = model.detect(integral.data(), 4, 4, scales, 1.f, 1);
    if (boxes.size() != 1 || boxes[0].r || boxes[0].c ||
        boxes[0].width != 3 || boxes[0].height != 3) return 2;
    if (!model.detect(integral.data(), 4, 4, scales, 1.f, 2).empty()) return 3;
    model.eps = -2.f;
    if (!model.detect(integral.data(), 4, 4, scales, 1.f, 1).empty()) return 4;
    try {
        const skimagexx::Cascade invalid("<cascade>");
        return 5;
    } catch (const skimagexx::CascadeXmlError&) {
        return 0;
    }
}
