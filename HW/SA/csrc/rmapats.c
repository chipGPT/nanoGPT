// file = 0; split type = patterns; threshold = 100000; total count = 0.
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "rmapats.h"

void  hsG_0__0 (struct dummyq_struct * I1353, EBLK  * I1348, U  I707);
void  hsG_0__0 (struct dummyq_struct * I1353, EBLK  * I1348, U  I707)
{
    U  I1613;
    U  I1614;
    U  I1615;
    struct futq * I1616;
    struct dummyq_struct * pQ = I1353;
    I1613 = ((U )vcs_clocks) + I707;
    I1615 = I1613 & ((1 << fHashTableSize) - 1);
    I1348->I752 = (EBLK  *)(-1);
    I1348->I753 = I1613;
    if (0 && rmaProfEvtProp) {
        vcs_simpSetEBlkEvtID(I1348);
    }
    if (I1613 < (U )vcs_clocks) {
        I1614 = ((U  *)&vcs_clocks)[1];
        sched_millenium(pQ, I1348, I1614 + 1, I1613);
    }
    else if ((peblkFutQ1Head != ((void *)0)) && (I707 == 1)) {
        I1348->I755 = (struct eblk *)peblkFutQ1Tail;
        peblkFutQ1Tail->I752 = I1348;
        peblkFutQ1Tail = I1348;
    }
    else if ((I1616 = pQ->I1256[I1615].I775)) {
        I1348->I755 = (struct eblk *)I1616->I773;
        I1616->I773->I752 = (RP )I1348;
        I1616->I773 = (RmaEblk  *)I1348;
    }
    else {
        sched_hsopt(pQ, I1348, I1613);
    }
}
#ifdef __cplusplus
extern "C" {
#endif
void SinitHsimPats(void);
#ifdef __cplusplus
}
#endif
