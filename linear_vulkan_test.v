module main

import vtl
import vtl.nn.layers

// Test Vulkan linear layer forward pass
fn test_linear_forward_vulkan() {
    // Create test data: x [2, 3], W [4, 3], b [4]
    mut x := vtl.zeros[f32]([2, 3])
    x.set([0, 0], 1.0)
    x.set([0, 1], 2.0)
    x.set([0, 2], 3.0)
    x.set([1, 0], 4.0)
    x.set([1, 1], 5.0)
    x.set([1, 2], 6.0)
    
    mut w := vtl.zeros[f32]([4, 3])
    w.set([0, 0], 0.1)
    w.set([0, 1], 0.2)
    w.set([0, 2], 0.3)
    w.set([1, 0], 0.4)
    w.set([1, 1], 0.5)
    w.set([1, 2], 0.6)
    w.set([2, 0], 0.7)
    w.set([2, 1], 0.8)
    w.set([2, 2], 0.9)
    w.set([3, 0], 1.0)
    w.set([3, 1], 1.1)
    w.set([3, 2], 1.2)
    
    mut b := vtl.zeros[f32]([4])
    b.set([0], 0.1)
    b.set([1], 0.2)
    b.set([2], 0.3)
    b.set([3], 0.4)
    
    // Test Vulkan path
    result := nn.layers.linear_forward_vulkan(x, w, b)!
    println('Linear forward result: ${result}')
    
    // Expected: x * W^T + b
    // x * W^T = [[1*0.1+2*0.4+3*0.7, 1*0.2+2*0.5+3*0.8, 1*0.3+2*0.6+3*0.9, 1*0.4+2*0.7+3*1.0],
    //            [4*0.1+5*0.4+6*0.7, 4*0.2+5*0.5+6*0.8, 4*0.3+5*0.6+6*0.9, 4*0.4+5*0.7+6*1.0]]
    //          = [[3.0, 3.6, 4.2, 4.8], [6.6, 8.1, 9.6, 11.1]]
    // + b = [[3.1, 3.8, 4.3, 4.9], [6.7, 8.3, 9.8, 11.5]]
    
    expected := vtl.zeros[f32]([2, 4])
    expected.set([0, 0], 3.1)
    expected.set([0, 1], 3.8)
    expected.set([0, 2], 4.3)
    expected.set([0, 3], 4.9)
    expected.set([1, 0], 6.7)
    expected.set([1, 1], 8.3)
    expected.set([1, 2], 9.8)
    expected.set([1, 3], 11.5)
    
    // Check if close enough (allowing for small floating point differences)
    res_arr := result.to_array()
    exp_arr := expected.to_array()
    assert res_arr.len == exp_arr.len
    for i in 0 .. res_arr.len {
        diff := (res_arr[i] - exp_arr[i]).abs()
        assert diff < 1e-5, 'Mismatch at index $i: got ${res_arr[i]}, expected ${exp_arr[i]}'
    }
    
    println('✓ Linear forward test passed')
    
    // Cleanup
    x.release()
    w.release()
    b.release()
    result.release()
}

// Test ReLU activation
fn test_relu_forward_vulkan() {
    mut x := vtl.zeros[f32]([4])
    x.set([0], -2.0)
    x.set([1], -0.5)
    x.set([2], 1.0)
    x.set([3], 2.0)
    result := nn.layers.relu_forward_vulkan(x)!
    println('ReLU result: ${result}')
    
    expected := vtl.zeros[f32]([4])
    expected.set([0], 0.0)
    expected.set([1], 0.0)
    expected.set([2], 1.0)
    expected.set([3], 2.0)
    
    res_arr := result.to_array()
    exp_arr := expected.to_array()
    assert res_arr.len == exp_arr.len
    for i in 0 .. res_arr.len {
        diff := (res_arr[i] - exp_arr[i]).abs()
        assert diff < 1e-5
    }
    
    println('✓ ReLU forward test passed')
    
    x.release()
    result.release()
}

// Test Sigmoid activation
fn test_sigmoid_forward_vulkan() {
    mut x := vtl.zeros[f32]([2])
    x.set([0], 0.0)
    x.set([1], 1.0)
    result := nn.layers.sigmoid_forward_vulkan(x)!
    println('Sigmoid result: ${result}')
    
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731
    expected := vtl.zeros[f32]([2])
    expected.set([0], 0.5)
    expected.set([1], 0.731)
    
    res_arr := result.to_array()
    exp_arr := expected.to_array()
    assert res_arr.len == exp_arr.len
    for i in 0 .. res_arr.len {
        diff := (res_arr[i] - exp_arr[i]).abs()
        assert diff < 1e-2  // Looser tolerance for sigmoid
    }
    
    println('✓ Sigmoid forward test passed')
    
    x.release()
    result.release()
}

fn main() {
    println('Testing Vulkan NN layers...')
    test_linear_forward_vulkan()
    test_relu_forward_vulkan()
    test_sigmoid_forward_vulkan()
    println('All tests passed!')
}