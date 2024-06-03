// Simple FIFO with parametrizable depth and width

module sync_fifo_data #(
    parameter SIZE = 16,
    parameter WIDTH = 32,
    parameter ALERT_DEPTH = 3
) (
    input                    clock, rstn,
    input                    wr_en,
    input                    rd_en,
    input  [WIDTH-1:0]       wr_data,
    output logic             wr_valid,
    output logic             rd_valid,
    output logic [WIDTH-1:0] rd_data,
    output logic             almost_full,
    output logic             full,
    output logic             empty
);
    logic [SIZE-1:0] [WIDTH-1:0] mem;
    logic [$clog2(SIZE+1)-1:0] head, next_head;
    logic [$clog2(SIZE+1)-1:0] tail, next_tail;

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
                next_tail=0;
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
            almost_full=((head-tail)==ALERT_DEPTH);
        else
            almost_full=((tail-head)==SIZE-ALERT_DEPTH);
    end

    assign rd_data = mem[head];

    always_ff @(posedge clock or negedge rstn) begin
        if (~rstn) begin
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
