`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 31.03.2022 07:11:00
// Design Name: 
// Module Name: convo4
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


module convo4(
input reg go,
    input clock,
    input reset,
    input [34:0] bias [15:0],
    input [34:0] in_raw [7:0][15:0][15:0],
    input wire [34:0] filter [127:0][2:0][2:0],
    //input wire [7:0] filter2 [2:0][2:0],
//    input [3:0] stride,
    output reg [34:0] conv_out [15:0][7:0][7:0],
    output reg flag
    );
    
parameter stride=2;
parameter m=18;
parameter k=2;
parameter d=16*8; //no of output channels
parameter id=8; //no of input channels
parameter pad=1;

//reg [7:0] filter [2:0][2:0];
reg [34:0] window [7:0][2:0][2:0];
reg [34:0] kernel [127:0][2:0][2:0];
reg [34:0] in [7:0][17:0][17:0];
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



task add_pad(input [34:0] in_raw [7:0][15:0][15:0],id, output [34:0] in [7:0][17:0][17:0]);
    for(int w=0;w<id;w++)
    for (int i=0;i<=17;i++)
    for(int j=0;j<=17;j++) begin
    if (i==0 || i==17|| j==0 || j==17)
    in[w][i][j]=0;
    else
    in[w][i][j]=in_raw[w][i-1][j-1];   
    end
    
endtask

task flip_filter(input [34:0] filter [127:0][2:0][2:0], k,d, output [34:0] flip [127:0][2:0][2:0]);
    for(int w = 0; w<d; w++) begin
    for (int i=0;i<=k;i+=1)begin
        for (int j=0;j<=k;j+=1) begin
            flip[w][i][j]=filter[w][k-i][k-j];
//                $display ("filter is %d",filter[k-i][k-j]);
     end
     end
     end
endtask

task get_window(input [2:0] start_r, input [2:0] start_c, output [34:0] window [7:0][2:0][2:0], input [34:0] in [7:0][17:0][17:0], k,id);
automatic int wi=0, wj=0;
for(int w = 0; w < id; w++) begin
for (int i=start_r,wi=0;i<=start_r+k;i+=1,wi+=1) begin
    for (int j=start_c,wj=0;j<=start_c+k;j+=1,wj+=1) begin
        window[w][wi][wj]=in[w][i][j];
        end
        end
end
endtask

task calculator (input [34:0] window [7:0][2:0][2:0], input [34:0] kernel [127:0][2:0][2:0], start_d, k, output [34:0] mult);
int win;
mult=0;
for (int i=0;i<=k;i+=1)
for(int j=0;j<=k;j+=1) begin
for (int w=0;w<16;w+=1) begin
if(window[w][i][j] && kernel[start_d+w][i][j]) 
qadd(kernel[start_d+w][i][j],mult,mult);
//$display ("%d %d ",window[0][i][j], mult); 
end
end
endtask



always @(negedge clock) begin
//$display("CONVO4");
flag=0;
if (go) begin
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
//$display ("inp %d ",in_raw[1][2][3]);
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

$display("CONVO4:");
for (int w=0,ind1=0;w<d;w+=8,ind1++) begin
$display("Layer %d",ind1);
for (int i=0,ind2=0;i<m-k;i+=stride,ind2++)begin
for (int j=0,ind3=0;j<m-k;j+=stride,ind3++) begin
get_window(i,j,window,in,k,id);
calculator(window, kernel, w, k, mult);
conv_out[ind1][ind2][ind3]=0;
if (w==0)
conv_out[ind1][ind2][ind3]=mult+bias[0];
else if (w==8)
conv_out[ind1][ind2][ind3]=mult+bias[1];
else if (w==16)
conv_out[ind1][ind2][ind3]=mult+bias[2];
else if (w==24)
conv_out[ind1][ind2][ind3]=mult+bias[3];
else if (w==32)
conv_out[ind1][ind2][ind3]=mult+bias[4];
else if (w==40)
conv_out[ind1][ind2][ind3]=mult+bias[5];
else if (w==48)
conv_out[ind1][ind2][ind3]=mult+bias[6];
else if (w==56)
conv_out[ind1][ind2][ind3]=mult+bias[7];
else if (w==64)
conv_out[ind1][ind2][ind3]=mult+bias[8];
else if (w==72)
conv_out[ind1][ind2][ind3]=mult+bias[9];
else if (w==80)
conv_out[ind1][ind2][ind3]=mult+bias[10];
else if (w==88)
conv_out[ind1][ind2][ind3]=mult+bias[11];
else if (w==96)
conv_out[ind1][ind2][ind3]=mult+bias[12];
else if (w==104)
conv_out[ind1][ind2][ind3]=mult+bias[13];
else if (w==112)
conv_out[ind1][ind2][ind3]=mult+bias[14];
else if (w==120)
conv_out[ind1][ind2][ind3]=mult+bias[15];


//$display ("%d", conv_out[ind1][ind2][ind3] ); 

end
end
end
go=0;
flag=1;
end
end
endmodule