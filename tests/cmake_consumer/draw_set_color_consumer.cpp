#include <skimagexx/draw_set_color.hpp>
#include <skimagexx/image.hpp>
#include <cassert>
#include <iostream>
#include <cmath>
int main(){using namespace skimagexx;using namespace skimagexx::detail;auto old=get_num_threads();int checks=0;
for(int workers:{1,4}){set_num_threads(workers);
float f32=.5f;double f32color=.25,f32alpha=.5;std::intptr_t f32zero=0;SetColorView f32view{&f32,{1,1,1},{4,4,0},&f32zero,&f32zero,1,&f32color,&f32alpha};f32view.image_float32=true;assert(draw_set_color(f32view,get_num_threads())&&f32==.375f);++checks;

double pixel=-0.,coefficient=0.,opacity=-1.;std::intptr_t zero=0;SetColorView stage{&pixel,{1,1,1},{8,8,0},&zero,&zero,1,&coefficient,&opacity};stage.integer_product=true;assert(draw_set_color(stage,get_num_threads())&&!std::signbit(pixel));++checks;
pixel=-0.;stage.integer_product=false;assert(draw_set_color(stage,workers)&&std::signbit(pixel));++checks;
double image[6]={.5,.75,1.,.25,.125,0.};std::intptr_t r[3]={0,0,1},c[3]={0,0,1};double alpha[3]={.25,.5,1.};double color[1]={.25};SetColorView v{image,{2,3,1},{24,8,0},r,c,3,color,alpha};assert(draw_set_color(v,get_num_threads()));assert(image[0]==.375&&image[4]==.25);checks+=3;
// Aliased color and image must be read before any scatter.
image[0]=.5;image[1]=.75;v.color=&image[0];v.count=2;c[0]=0;c[1]=1;r[1]=0;alpha[0]=1.;alpha[1]=1.;assert(draw_set_color(v,workers));assert(image[0]==.5&&image[1]==.5);checks+=3;
// Signed image strides and clipped coordinates, no out-of-bounds reads.
v.image=image+5;v.strides={-24,-8,0};v.color=color;r[0]=-1;r[1]=0;c[1]=1;v.count=2;assert(draw_set_color(v,workers));assert(image[4]==.25);checks+=2;
}
set_num_threads(old);assert(get_num_threads()==1);std::cout<<checks<<" installed-client controls PASS default1 restored\n";}
