`timescale 1ns / 1ps

  module nrn(
    input reg flag_in,
    input clock,
    input reset,
    input reg [34:0] vmembrane,
    input [34:0] in,
    output reg [34:0]out,
    output reg [34:0] mem_out,
    output reg flag_out
    );
      
    reg [34:0] beta=35'b00010110011001100110011001100110011;
    reg [34:0] vrest = 35'h4;
    reg [34:0] vleak = 35'h1;
//    reg [34:0] vth = 35'b00000110011001100110011001100110011;
reg [34:0] vth = 35'b0;
    reg [34:0] vmin=35'h1;
    //reg [34:0] vmembrane = 4'h0;
    reg [34:0] vrise=2;
    reg [1:0] delay = 0;
    parameter refrac_prd=2;
    parameter Q = 32;
	parameter N = 35;
	
	 reg [N-1:0]	temp ;
//	 reg [N-1:0]	i_multiplier;
//	 reg [N-1:0]	o_result;
	 reg [2*N-1:0]	r_result;		//	Multiplication by 2 values of N bits requires a 
											//		register that is N+N = 2N deep...
  
  
  task qadd(input [N-1:0] a, input [N-1:0] b, output [N-1:0] res);
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

  task qmult(input [N-1:0] i_multiplier, output	[N-1:0]	o_result);

		r_result = beta[N-2:0] * i_multiplier[N-2:0];	//	Removing the sign bits from the multiply - that 
		o_result[N-1] = beta[N-1] ^ i_multiplier[N-1];	//		which is the XOR of the input sign bits...  (you do the truth table...)
		o_result[N-2:0] = r_result[N-2+Q:Q];								//	And we also need to push the proper N bits of result up to 
endtask
 
 //////////////////////////////////////////////////////////////////////////
  
    always @(flag_in) begin
    out=0;
    mem_out=0;
    temp=0;
//    mem1 <= vmembrane;
//$display("FLAG %d",flag_in);
    if (flag_in) 
    begin
//    0.10110011001100110011001100110011
//    $display("vmemin %b %b",vmembrane, in);
//    if (reset) begin
//    vmembrane = vrest;
//    spike = 0;
//    out<=0;
//    delay<=0;
    
//    end
//    else begin
//    $display("in neuron %d",in);
    if (vmembrane[N-2:0]<=vth || vmembrane[N-1]==1) begin
    delay=0;
    
    qmult(vmembrane,temp);
    qadd(temp,in,vmembrane);
    
//    vmembrane = beta*vmembrane + in;
        if (vmembrane[N-1]==0 && vmembrane[N-2:0] > vth[N-2:0]) begin
          temp=vth;
          temp[N-2:0]=~temp[N-2:0]+35'b00000000000000000000000000000000001;
          temp[N-1]=1;
          qadd(vmembrane,temp,vmembrane);
//          vmembrane = vmembrane-vth;
          out = 1;
//          $display("s %d",out);
//$display("in nrn vth=%b vmem=%b",vth,vmembrane);
//    $display("in nrn temp=%b in=%b",temp,in);
        end
//        if (vmembrane <vrest)
//            vmembrane<=vrest;
    end
   
   else if (vmembrane[N-1]==0 && vmembrane[N-2:0] > vth[N-2:0]) begin
   out=1;
   qmult(vmembrane,vmembrane);
//   vmembrane = beta*vmembrane;
//   $display("still sort");
//$display("s %d",out);
   end
   
//    else if (vmembrane < vrest) begin
//      vmembrane = vmembrane + vrise;
////      delay=delay+1;
//      out=0;
////      $display("here also");
//    end
        flag_out=1;
        mem_out=vmembrane;
//        $display("output %f",$itor(mem_out[N-2:0]*(2.0**-32)));

    end
//    flag_in=0;
    

  end

endmodule
