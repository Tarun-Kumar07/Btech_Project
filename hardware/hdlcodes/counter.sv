`timescale 1ns / 1ps
module counter(input reg go,
               input clock,
               input [34:0] flat_out [10:0], 
               input reg [7:0] pred_in [10:0],
               output reg [7:0] pred_out [10:0],
               output reg flag
               );
               
               
always @(negedge clock) begin
    flag=0;
    if (go) begin
//    $display("in counter");
$display("CONVO6 counter state");
    for (int i=0;i<11;i++) begin
    if (flat_out[i]==1) 
    pred_in[i]=pred_in[i]+1;
    

//    $display("counter state");
   
    end
    pred_out=pred_in;
    flag=1;
    go=0;
    for(int i=0;i<11;i++)
    $display("counting %b %d", pred_out[i], flat_out[i]);
//    $display("7\n54\n8\n2\n11\n23\n5\n13\n7\n14\n6");
    end
end
endmodule
