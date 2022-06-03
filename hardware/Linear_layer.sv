`timescale 1ns / 1ps

module Linear_layer( input reg go, input clock, 
                    input [34:0] in_raw [15:0][7:0][7:0],
                    input [34:0] filter[10:0][1023:0], 
                    input [34:0] bias[10:0],
                    output reg [34:0] out [10:0], 
                    output reg flag
                    );
    
    reg [34:0] in [1023:0];
    int w = 0;
    
    always @(negedge clock) begin
    flag=0;
    if (go) begin
    $display("CONVO 5 FINAL LAYER");

    w = 0;
    for(int i = 0; i < 16; i++) begin
    for(int j = 0; j < 8; j++)  begin
    for(int k = 0; k < 8; k++)
    in[w] = in_raw[i][j][k];
    w++;
    
    end
    end
    
//    for(int i = 0; i < 11; i++)
//    out[i] = 0;
    
    for(int i = 0; i < 11; i++) begin
    out[i]=bias[i];
    for(int j = 0; j < 1024; j++) 
    if(in[j]==1)
    out[i] += filter[i][j];
    $display("last %b",out[i]);
    end
    go=0;
    flag=1;
    end
    end
endmodule
