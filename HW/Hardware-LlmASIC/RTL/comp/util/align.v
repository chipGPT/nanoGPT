// +FHDR========================================================================
//  License:
//
// =============================================================================
//  File Name:      align.v
//                  Shiwei Liu (liushiwei@google.com)
//  Organization:   Space Beaker Team, Google Research
//  Description:
//      Register file with different input and output width
// -FHDR========================================================================

// =============================================================================
// Series to Parallel

module align_s2p #(
    parameter   IDATA_BIT = 64,
    parameter   ODATA_BIT = 256
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Data Signals
    input       [IDATA_BIT-1:0] idata,
    input                       idata_valid,
    output  reg [ODATA_BIT-1:0] odata,
    output  reg                 odata_valid
);

    localparam  REG_NUM = ODATA_BIT / IDATA_BIT;
    localparam  ADDR_BIT = $clog2(REG_NUM);

    // 1. Register file / buffer
    reg     [IDATA_BIT-1:0] regfile [0:REG_NUM-1];
    reg     [ADDR_BIT-1:0]  regfile_addr;           

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            regfile_addr <= 'd0;
        end
        else if (idata_valid) begin
            regfile_addr <= regfile_addr + 1'b1;
        end
    end

    always @(posedge clk) begin
        if (idata_valid) begin
            regfile[regfile_addr] <= idata;
        end
    end

    // 2. Output
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata_valid <= 1'b0;
        end
        else begin
            if (&regfile_addr && idata_valid) begin
                odata_valid <= 1'b1;
            end
            else begin
                odata_valid <= 1'b0;
            end
        end
    end

    genvar i;
    generate
        for (i = 0; i < REG_NUM; i = i + 1) begin:gen_pal
            always @(*) begin
                odata[i*IDATA_BIT+:IDATA_BIT] = regfile[i];
            end
        end
    endgenerate
    
endmodule

// =============================================================================
// Parallel to Series

module align_p2s #(
    parameter   IDATA_BIT = 256
    parameter   ODATA_BIT = 64
)(
    // Global Signals
    input                       clk,
    input                       rst,

    // Data Signals
    input       [IDATA_BIT-1:0] idata,
    input                       idata_valid,
    output  reg [ODATA_BIT-1:0] odata,
    output  reg                 odata_valid
);

    localparam  REG_NUM = IDATA_BIT / ODATA_BIT;
    localparam  ADDR_BIT = $clog2(REG_NUM);

    // 1. Register File / Buffer
    reg     [ODATA_BIT-1:0] regfile [0:REG_NUM-1];
    reg     [ADDR_BIT-1:0]  regfile_addr;
    reg                     regfile_valid;

    genvar i;
    generate
        for (i = 0; i < REG_NUM; i = i + 1) begin: gen_ser
            always @(posedge clk or posedge rst) begin
                if (rst) begin
                    regfile[i] <= 'd0;
                end
                else if (idata_valid) begin
                    regfile[i] <= idata[i*ODATA_BIT+:ODATA_BIT];
                end
            end
        end
    endgenerate

    // 2. FSM: segment counter
    parameter   REGFILE_IDLE  = 2'b01,
                REGFILE_VALID = 2'b10;
    reg     [1:0]   regfile_state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            regfile_state <= 'd0;
            regfile_valid <= 1'b0;
            regfile_addr  <= 'd0;
        end
        else begin
            case (regfile_state)
                REGFILE_IDLE: begin
                    if (idata_valid) begin
                        regfile_state <= REGFILE_VALID;
                        regfile_addr  <= 'd0;
                        regfile_valid <= 1'b1;
                    end
                end
                REGFILE_VALID: begin
                    if (regfile_addr == REG_NUM - 1'b1) begin
                        if (idata_valid) begin
                            regfile_state <= REGFILE_VALID;
                            regfile_addr  <= 'd0;
                            regfile_valid <= 1'b1;
                        end
                        else begin
                            regfile_state <= REGFILE_IDLE;
                            regfile_addr  <= 'd0;
                            regfile_valid <= 1'b0;
                        end
                    end
                    else begin
                        regfile_addr <= regfile_addr + 1'b1;
                    end
                end
                default: begin
                    regfile_state <= REGFILE_IDLE;
                    regfile_addr  <= 'd0;
                    regfile_valid <= 1'b0;
                end
            endcase
        end
    end

    // 3. Output
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata <= 'd0;
        end
        else if (regfile_valid) begin
            odata <= regfile[regfile_addr];
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            odata_valid <= 1'b0;
        end
        else begin
            odata_valid <= regfile_valid;
        end
    end

endmodule