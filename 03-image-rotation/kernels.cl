__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_FILTER_NEAREST |
    CLK_ADDRESS_CLAMP;

__kernel void image_rotate(
    __read_only  image2d_t src,
    __write_only image2d_t dst,
    float angle
)
{
    printf("hello\n");
    int width  = get_image_width(src);
    int height = get_image_width(src);
    int const x = get_global_id(0);
    int const y = get_global_id(1);
    float sina = sin(angle);
    float cosa = cos(angle);

    int hwidth  = width / 2;
    int hheight = width / 2;
    int xt = x - hwidth;
    int yt = y - hheight;

    float2 read;
    read.x = cosa * xt - sina * yt + hwidth;
    read.y = sina * xt + cosa * yt + hheight;

    float4 value = read_imagef(src, sampler, read);

    // if (x == 0 && y == 0) printf("f4 = %2.2v4hlf\n", value);

    write_imagef(dst, (int2)(x, y), value);
}

