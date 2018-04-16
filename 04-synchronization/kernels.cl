typedef int value_type;

void kernel add(global value_type const* a, global value_type const* b, global value_type* c)
{
    auto id = get_global_id(0);
    c[id] = a[id] + b[id];
}

void kernel mul_minus(global value_type const* a, global value_type const* b, global value_type* c)
{
    auto id = get_global_id(0);
    c[id] = c[id] * a[id] - b[id];
}

