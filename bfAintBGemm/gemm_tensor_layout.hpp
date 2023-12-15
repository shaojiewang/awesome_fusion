#pragma once

namespace gemm_layout {

struct BaseTensorLayout
{
};

namespace gemm {

struct RowMajor : public BaseTensorLayout
{
    static constexpr const char* name = "RowMajor";
};

struct ColumnMajor : public BaseTensorLayout
{
    static constexpr const char* name = "ColumnMajor";
};
} // namespace gemm

template <
    typename Layout,
    typename std::enable_if<std::is_base_of<BaseTensorLayout, Layout>::value, bool>::type = false>
std::ostream& operator<<(std::ostream& os, const Layout&){
    os << Layout::name;
    return os;
}

}
