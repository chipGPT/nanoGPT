// Simple FIFO with parametrizable depth and width

module sync_fifo #(
    parameter SIZE = 16,
    parameter WIDTH = 32,
    parameter ALERT_DEPTH = 3,
    parameter   IN_DEPTH = 6
) (
    input                    clock, reset,
    input                    wr_en,
    input                    rd_en,
    input  [IN_DEPTH-1:0][WIDTH-1:0] wr_data,
    output logic             wr_valid,
    output logic             rd_valid,
    output logic [IN_DEPTH-1:0][WIDTH-1:0] rd_data,
    output logic             almost_full,
    output logic             full,
    output logic             empty
);
    // LAB5 TODO: Make the FIFO, see other TODOs below
    // Some things you will need to do:
    // - Define the sizes for your head (read) and tail (write) pointer
    // - Increment the pointers if needed (hint: try using the modulo operator: '%') //tgc question
    // - Write to the tail when wr_en == 1 and the fifo isn't full
    // - Read from the head when rd_en == 1 and the fifo isn't empty


    logic [SIZE-1:0] [WIDTH-1:0] mem; // LAB5 TODO: name this


    // LAB5 TODO: how wide is a pointer to SIZE elements?
    logic [$clog2(SIZE+1)-1:0] head, next_head; //tgc question, what's the point, clog2 has ceiling
    logic [$clog2(SIZE+1)-1:0] tail, next_tail;


    // LAB5 TODO: Use one of three ways to track if full/empty:
    //  1. (easiest) Keep a count of the number of entries
    //  2. (medium)  Use a valid bit for each entry
    //  3. (hardest) Use head == tail and keep a state of empty vs. full in always_ff
    //               This is hardest because you also need to track almost_full
    logic head_val;
    logic tail_val;
    logic rd_valid_d;
    logic wr_valid_d;
    assign wr_valid= wr_en && (!full || rd_en);
    assign rd_valid= rd_en && !empty ;


    logic tail_overflow;
    logic head_overflow;
    assign tail_overflow = (tail == SIZE-1);
    assign head_overflow = (head == SIZE-1);

    always_comb begin
        if(wr_valid) begin
            if(tail_overflow)
                next_tail=0;//%8
            else
                next_tail=tail+1;
        end
        else begin
            next_tail=tail;
        end

        if(rd_valid) begin
            if(head_overflow)
                next_head=0;
            else
                next_head=head+1;
        end
        else begin
            next_head=head;
        end
    end

    assign empty=(head==tail)&&(head_val==tail_val);
    assign full =(head==tail)&&(head_val^tail_val);

    always_comb begin
        if(head_val^tail_val)
            almost_full=((head-tail)==ALERT_DEPTH); //tgc question, why almost full for superscalar?
        else
            almost_full=((tail-head)==SIZE-ALERT_DEPTH);
    end

    // LAB5 TODO: Output read data from the head combinationally
    //            (doing this correctly in always_ff is somewhat difficult)
    assign rd_data = mem[head];


    always_ff @(posedge clock) begin
        if (reset) begin
            // LAB5 TODO: Initialize state variables
            head_val<=0;
            tail_val<=0;
            head<=0;
            tail<=0;
            rd_valid_d<=0;
            wr_valid_d<=0;
            for(int i=0;i<SIZE;i++)
                mem[i]<=0;
        end else begin
            rd_valid_d<=rd_valid;
            wr_valid_d<=wr_valid;
            // LAB5 TODO: Update on each cycle (use wr_data somewhere in here)
            if(head_overflow && rd_valid_d)
                head_val<=!head_val;
            if(tail_overflow && wr_valid_d)
                tail_val<=!tail_val;
            tail<=next_tail;
            head<=next_head;

            if(wr_valid)
                mem[tail]<=wr_data;
        end
    end

endmodule
