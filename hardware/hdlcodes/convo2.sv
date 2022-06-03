`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 31.03.2022 05:35:23
// Design Name: 
// Module Name: convo2
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


module convo2(
    input reg go,
    input clock,
    input reset,
    input [34:0] bias [7:0],
    input [34:0] in_raw [3:0][63:0][63:0],
    input wire [34:0] filter [31:0][4:0][4:0],
    //input wire [7:0] filter2 [2:0][2:0],
//    input [3:0] stride,
    output reg [34:0] conv_out [7:0][31:0][31:0],
    output reg flag
    );
    
parameter stride=2;
parameter m=68;
parameter k=4;
parameter d=32; //no of output channels
parameter id=4; //no of input channels
parameter pad=2;

//reg [7:0] filter [2:0][2:0];
reg [34:0] window [3:0][4:0][4:0];
reg [34:0] kernel [31:0][4:0][4:0];
reg [34:0] in [3:0][67:0][67:0];
//reg [7:0] kernel2 [2:0][2:0];
reg [34:0] mult;
//reg [7:0] conv_out [3:0][63:0][63:0];
integer row,col,ind1,ind2,ind3;
integer i=0,j=0;
reg [2:0] start_r;
reg [2:0] start_c ;
reg [15:0] temp1,temp2;
reg [7:0] temp3;
parameter Q = 32;
parameter N = 35;


task qadd(
    input [N-1:0] a,
    input [N-1:0] b,
    output [N-1:0] res
    );
//  reg [N-1:0] res;

//assign c = res;

	// both negative or both positive
	if(a[N-1] == b[N-1]) begin						//	Since they have the same sign, absolute magnitude increases
		res[N-2:0] = a[N-2:0] + b[N-2:0];		//		So we just add the two numbers
		res[N-1] = a[N-1];							//		and set the sign appropriately...  Doesn't matter which one we use, 
															//		they both have the same sign
															//	Do the sign last, on the off-chance there was an overflow...  
		end												//		Not doing any error checking on this...
	//	one of them is negative...
	else if(a[N-1] == 0 && b[N-1] == 1) begin		//	subtract a-b
		if( a[N-2:0] > b[N-2:0] ) begin					//	if a is greater than b,
			res[N-2:0] = a[N-2:0] + b[N-2:0];			//		then just subtract b from a
			res[N-1] = 0;										//		and manually set the sign to positive
			end
		else begin												//	if a is less than b,
			res[N-2:0] = b[N-2:0] + a[N-2:0];			//		we'll actually subtract a from b to avoid a 2's complement answer
			if (res[N-2:0] == 0)
				res[N-1] = 0;										//		I don't like negative zero....
			else
				res[N-1] = 1;										//		and manually set the sign to negative
			end
		end
	else begin												//	subtract b-a (a negative, b positive)
		if( a[N-2:0] > b[N-2:0] ) begin					//	if a is greater than b,
			res[N-2:0] = a[N-2:0] + b[N-2:0];			//		we'll actually subtract b from a to avoid a 2's complement answer
			if (res[N-2:0] == 0)
				res[N-1] = 0;										//		I don't like negative zero....
			else
				res[N-1] = 1;										//		and manually set the sign to negative
			end
		else begin												//	if a is less than b,
			res[N-2:0] = b[N-2:0] + a[N-2:0];			//		then just subtract a from b
			res[N-1] = 0;										//		and manually set the sign to positive
			end
		end
//		$display("add: %d %d %d",a,b,res);
endtask    


task add_pad(input [34:0] in_raw [3:0][63:0][63:0],id, output [34:0] in [3:0][67:0][67:0]);
    for(int w=0;w<id;w++)
    for (int i=0;i<=67;i++)
    for(int j=0;j<=67;j++) begin
    if (i==0 || i==1 ||i==66 || i==67|| j==0 || j==1 || j==66 || j==67)
    in[w][i][j]=0;
    else
    in[w][i][j]=in_raw[w][i-2][j-2];   
    end
    
endtask

task flip_filter(input [34:0] filter [31:0][4:0][4:0], k,d, output [34:0] flip [31:0][4:0][4:0]);
    for(int w = 0; w<d; w++) begin
    for (int i=0;i<=k;i+=1)begin
        for (int j=0;j<=k;j+=1) begin
            flip[w][i][j]=filter[w][k-i][k-j];
//                $display ("filter is %d",filter[k-i][k-j]);
     end
     end
     end
endtask

task get_window(input [2:0] start_r, input [2:0] start_c, output [34:0] window [3:0][4:0][4:0], input [34:0] in [3:0][67:0][67:0], k,id);
automatic int wi=0, wj=0;
for(int w = 0; w < id; w++) begin
for (int i=start_r,wi=0;i<=start_r+k;i+=1,wi+=1) begin
    for (int j=start_c,wj=0;j<=start_c+k;j+=1,wj+=1) begin
        window[w][wi][wj]=in[w][i][j];
        end
        end
end
endtask

task calculator (input [34:0] window [3:0][4:0][4:0], input [34:0] kernel [31:0][4:0][4:0], start_d, k, output [34:0] mult);
int win;
mult=0;
for (int i=0;i<=k;i+=1)
for(int j=0;j<=k;j+=1) begin
for (int w=0;w<4;w+=1) begin
if(window[w][i][j] && kernel[start_d+w][i][j]) 
qadd(kernel[start_d+w][i][j],mult,mult);
//mult+=window[0][i][j]*kernel[start_d][i][j] + window[1][i][j]*kernel[start_d+1][i][j];
//qadd(kernel[start_d][i][j], kernel[start_d+1][i][j],temp3);
//qadd(temp3,mult,mult);
//$display ("%d %d ",window[0][i][j], mult); 
end
end
endtask



always @(negedge clock) begin
flag=0;
//$display ("CONVO2");
if (go) begin
go=0;
flip_filter(filter,k,d,kernel);
//$display("bias %d %d %d %d",bias[0],bias[1],bias[2],bias[3]);
////flip_filter(filter2,k,kernel2);
//$display ("Input Kernel1:");
//for (int w = 0; w < d; w++) begin
//$display("Layer%d",w);
//for(int i = 0; i<=k; i++)
//$display ("%d %d %d ", filter[w][i][0], filter[w][i][1], filter[w][i][2], filter[w][i][3], filter[w][i][4]); 
//end

//$display ("Flipped Kernel1:");
//for (int i = 0; i < d; i++) begin
//$display("Layer%d",i);
//for (int j=0;j<=k;j++)
//$display ("%d %d %d ", kernel[i][j][0], kernel[i][j][1], kernel[i][j][2], kernel[i][j][3], kernel[i][j][4]); 
//end
//////get_window(0,0,window,in, 2);

//$display ("Input matrix:");
//for(int i = 0; i <2; i++) begin
//$display("Layer%d",i);
//for (int j=0;j<m-4;j++)
//for (int w=0;w<m-4;w++)
$display ("inp %d ",in_raw[1][2][3]);
//end

add_pad(in_raw,id,in);
//$display ("Padded matrix:");
//for(int i = 0; i <2; i++) begin
//$display("Layer%d",i);
//for (int j=0;j<m;j++) begin
//$display("Row%d",j);
//for (int w=0;w<m;w++)
//$display ("%d ",in[i][j][w]);
//end
//end

$display("CONVO2:");
for (int w=0,ind1=0;w<d;w+=4,ind1++) begin
$display("Layer %d",ind1);
for (int i=0,ind2=0;i<m-k;i+=stride,ind2++)begin
for (int j=0,ind3=0;j<m-k;j+=stride,ind3++) begin
get_window(i,j,window,in,k,id);
calculator(window, kernel, w, k, mult);
conv_out[ind1][ind2][ind3]=0;
if (w==0)
conv_out[ind1][ind2][ind3]=mult+bias[0];
else if (w==4)
conv_out[ind1][ind2][ind3]=mult+bias[1];
else if (w==8)
conv_out[ind1][ind2][ind3]=mult+bias[2];
else if (w==12)
conv_out[ind1][ind2][ind3]=mult+bias[3];
else if (w==16)
conv_out[ind1][ind2][ind3]=mult+bias[4];
else if (w==20)
conv_out[ind1][ind2][ind3]=mult+bias[5];
else if (w==24)
conv_out[ind1][ind2][ind3]=mult+bias[6];
else if (w==28)
conv_out[ind1][ind2][ind3]=mult+bias[7];


//$display ("%d", conv_out[ind1][ind2][ind3] ); 

end
end
end
flag=1;
end
end
endmodule
