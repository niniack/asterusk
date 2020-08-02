mod test;

#[macro_use]
pub mod utils;
pub mod edge_detection;
pub mod fftnd;
pub mod kernel_conv;
pub mod max_pool;

pub mod prelude {
    pub use crate::edge_detection::*;
    pub use crate::fftnd::*;
    pub use crate::kernel_conv::*;
    pub use crate::max_pool::*;
    pub use crate::utils::*;
}
