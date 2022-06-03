`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 25.03.2022 15:32:01
// Design Name: 
// Module Name: layer1
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module layer1(input reg go,
    input clock,
    input reset,
    input [7:0] pred_in [10:0],
    input [7:0] bias_ad,
    input [7:0] bias2_ad,
    input [7:0] bias3_ad,
    input [7:0] bias4_ad,
    input [7:0] bias5_ad,
    input [15:0] in_ad,
    input [7:0] filter_ad,
    input [15:0] filter2_ad,
    input [15:0] filter3_ad,
    input [15:0] filter4_ad,
    input [15:0] filter5_ad,
    output [7:0] pred_out [10:0]
    );
    
//    initial $display("called %d",vmem_in[0][0][0]);
//    reg [34:0] filter2 [31:0][4:0][4:0];
//    reg [34:0] filter3 [63:0][2:0][2:0];
//    reg [34:0] filter4 [127:0][2:0][2:0];
//    reg [34:0] filter5 [10:0][1023:0];
    reg [34:0] bias [3:0];
    reg [34:0] bias2 [7:0];
    reg [34:0] bias3 [7:0];
    reg [34:0] bias4 [15:0];
    reg [34:0] bias5 [10:0];
    reg [24:0] in_raw [1:0][127:0][127:0];
    reg  [34:0] filter [7:0][4:0][4:0];
    reg  [34:0] filter2 [31:0][4:0][4:0];
    reg  [34:0] filter3 [63:0][2:0][2:0];
    reg  [34:0] filter4 [127:0][2:0][2:0];
    reg  [34:0] filter5 [10:0][1023:0];
    reg  [34:0] vmem_in [3:0][63:0][63:0];
    reg  [34:0] vmem_in2 [7:0][31:0][31:0];
    reg  [34:0] vmem_in3 [7:0][15:0][15:0];
    reg  [34:0] vmem_in4 [15:0][7:0][7:0];
    reg  [34:0] vmem_in5 [10:0];
    reg [34:0] vmem_out [3:0][63:0][63:0];
    reg [34:0] vmem_out2 [7:0][31:0][31:0];
    reg [34:0] vmem_out3 [7:0][15:0][15:0];
    reg [34:0] vmem_out4 [15:0][7:0][7:0];
    reg [34:0] vmem_out5 [10:0];

    reg [34:0]out [3:0][63:0][63:0];
    reg [34:0] conv_out[3:0][63:0][63:0];
    reg [34:0] conv_out2[7:0][31:0][31:0];
    reg [34:0] out2[7:0][31:0][31:0];

    reg [34:0] conv_out3[7:0][15:0][15:0];
    reg [34:0] out3[7:0][15:0][15:0];

    reg [34:0] conv_out4[15:0][7:0][7:0];
    reg [34:0] out4[15:0][7:0][7:0]; 

    reg [34:0] lin[1023:0][10:0];
    reg [34:0] flat[10:0];
    reg [34:0] out5 [10:0];
    reg [10:0] flag;
    reg [10:0] pred; 
    
    
    //filler function
//    always@(negedge clock) 
//     bias [3:0];
//     bias2 [7:0];
//     bias3 [7:0];
//     bias4 [15:0];
//     bias5 [10:0];
//     in_raw [1:0][127:0][127:0];
//     filter [7:0][4:0][4:0];
//     filter2 [31:0][4:0][4:0];
//     filter3 [63:0][2:0][2:0];
//     filter4 [127:0][2:0][2:0];
//     filter5 [10:0][1023:0];
//     vmem_in [3:0][63:0][63:0];
//     vmem_in2 [7:0][31:0][31:0];
//     vmem_in3 [7:0][15:0][15:0];
//     vmem_in4 [15:0][7:0][7:0];
//     vmem_in5 [10:0];
//     vmem_out [3:0][63:0][63:0];
//     vmem_out2 [7:0][31:0][31:0];
//     vmem_out3 [7:0][15:0][15:0];
//     vmem_out4 [15:0][7:0][7:0];
//     vmem_out5 [10:0];
//    end


    convo l1(go,clock,reset,bias,in_raw,filter,conv_out,flag[0]);
//    begin
//    if (flag[0]) begin
//    $display("op");
//    for(int i=0;i<4;i++)
//        for(int j=0;j<64;j++)
//        for(int k=0;k<64;k++)
//        $display("%d",conv_out[i][j][k]);
//        $display("mem");
//     end
//     end

////////////////WANTED]>
    generate
    genvar i,j,k;
        for(i=0;i<4;i++)
        for(j=0;j<64;j++)
        for(k=0;k<64;k++) 
        begin
        nrn n(flag[0],clock, reset,vmem_in [i][j][k],conv_out[i][j][k],out[i][j][k],vmem_out[i][j][k],flag[1]);
//        assign done=done+1;
        end
       // assign done=1;
    endgenerate
///////////////////BAS[<    
    
////    always@(flag[1])
////    begin
////    if (done) begin
////    $display("SPIKES");
////    for(int i=0;i<4;i++)
////        for(int j=0;j<64;j++)
////        for(int k=0;k<64;k++)
////        $display("%d",out[i][j][k]);
////     end
////end
   
//   //=============================================================================//
   
    convo2 l2(flag[1],clock,reset,bias2,out,filter2,conv_out2,flag[2]);
    generate
//    genvar i,j,k;
        for(i=0;i<8;i++)
        for(j=0;j<32;j++)
        for(k=0;k<32;k++) 
        begin
        nrn n2(flag[2],clock, reset,vmem_in2[i][j][k],conv_out2[i][j][k],out2[i][j][k],vmem_out2[i][j][k],flag[3]);
//        assign done=done+1;
        end
       // assign done=1;
    endgenerate
    
//    //=================================================================================//
    
    convo3 l3(flag[3],clock,reset,bias3,out2,filter3,conv_out3,flag[4]);
    generate
    //genvar i,j,k;
        for(i=0;i<8;i++)
        for(j=0;j<16;j++)
        for(k=0;k<16;k++) 
        begin
        nrn n3(flag[4],clock, reset,vmem_in3[i][j][k],conv_out3[i][j][k],out3[i][j][k],vmem_out3[i][j][k],flag[5]);
//        assign done=done+1;
        end
       // assign done=1;
    endgenerate
    
//    //=================================================================================//
    
    convo4 l4(flag[5],clock,reset,bias4,out3,filter4,conv_out4,flag[6]);
    generate
    //genvar i,j,k;
        for(i=0;i<16;i++)
        for(j=0;j<8;j++)
        for(k=0;k<8;k++) 
        begin
        nrn n4(flag[6],clock, reset,vmem_in4[i][j][k],conv_out4[i][j][k],out4[i][j][k],vmem_out4[i][j][k],flag[7]);
//        assign done=done+1;
        end
       // assign done=1;
    endgenerate

//    //=================================================================================//
   
    Linear_layer l5( flag[7], clock, out4, filter5, bias5,flat, flag[8]);
    generate
    //genvar i,j,k;
        for(i=0;i<11;i++)
        begin
        nrn n5(flag[8],clock, reset,vmem_in5[i],flat[i],out5[i],vmem_out5[i],flag[9]);
//        assign done=done+1;
        end
       // assign done=1;
    endgenerate
    
    counter cnt1 (flag[9], clock, out5, pred_in, pred_out, flag[10]);
endmodule
