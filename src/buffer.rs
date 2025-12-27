
use std::alloc::{alloc, dealloc, Layout};
use std::sync::{Arc, Mutex};

pub struct PageAlignedByteBuffer {
    data: Option<Arc<Mutex<Vec<u8>>>>,

    pointer: *mut u8,
    layout: Layout,
}

impl PageAlignedByteBuffer {
    pub fn new(buffer_size: usize) -> Self {
        let align = page_size::get();
        let layout = unsafe {

            Layout::from_size_align_unchecked(buffer_size, align)
        };
        let pointer = unsafe { alloc(layout) };
        assert!(!pointer.is_null(), "Allocation failed"); 

        let data = unsafe {
            Vec::from_raw_parts(pointer, buffer_size, buffer_size)
        };
        PageAlignedByteBuffer {
            data: Some(Arc::new(Mutex::new(data))),
            pointer,
            layout,
        }
    }

    pub fn get_buffer(&self) -> Arc<Mutex<Vec<u8>>> {
        self.data.as_ref().unwrap().clone()
    }
}

impl Drop for PageAlignedByteBuffer {
    fn drop(&mut self) {
        std::mem::forget(self.data.take().unwrap());

        unsafe {
            dealloc(self.pointer, self.layout);
        }
    }
}

unsafe impl Send for PageAlignedByteBuffer {}

#[cfg(test)]
mod buffer_tests {
    use super::PageAlignedByteBuffer;

    #[test]
    fn buffer_creation_destruction_test() {
        {
            let _test = PageAlignedByteBuffer::new(1024 * 1024);
        }
        assert!(true);
    }
}
