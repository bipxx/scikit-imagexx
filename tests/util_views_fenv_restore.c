/* Narrow retained Linux x86-64/glibc fenv_t restoration, not a project core.
 * Only trusted current-thread fegetenv snapshots from this verified ABI.
 * x87 stack must have been empty at the saved point; register DATA are not
 * represented by fenv_t. All exceptions must remain masked. No target calls.
 */
#include <fenv.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if !defined(__linux__) || !defined(__x86_64__) || !defined(__GLIBC__)
# error "This helper requires retained Linux x86-64 glibc"
#endif
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
# error "This helper requires retained little-endian ABI"
#endif
_Static_assert(sizeof(fenv_t)==32,"fenv_t ABI changed");
_Static_assert(offsetof(fenv_t,__control_word)==0,"control ABI changed");
_Static_assert(offsetof(fenv_t,__status_word)==4,"status ABI changed");
_Static_assert(offsetof(fenv_t,__tags)==8,"tags ABI changed");
_Static_assert(offsetof(fenv_t,__eip)==12,"instruction pointer ABI changed");
_Static_assert(offsetof(fenv_t,__data_offset)==20,"data pointer ABI changed");
_Static_assert(offsetof(fenv_t,__mxcsr)==28,"MXCSR ABI changed");
_Static_assert(sizeof(void*)==8,"pointer ABI changed");
#define PUBLIC __attribute__((visibility("default")))

/* Compact compile-time ABI description for the existing ctypes FEnv model. */
PUBLIC int reference_fenv_layout(uint32_t* words,size_t count) {
    if (!words || count!=8) return -1;
    const uint32_t layout[8]={sizeof(fenv_t),offsetof(fenv_t,__control_word),
        offsetof(fenv_t,__status_word),offsetof(fenv_t,__tags),
        offsetof(fenv_t,__eip),offsetof(fenv_t,__data_offset),
        offsetof(fenv_t,__mxcsr),sizeof(void*)};
    memcpy(words,layout,sizeof(layout));
    return 0;
}

PUBLIC int reference_fenv_restore(const void* snapshot,size_t bytes) {
    if (!snapshot || bytes!=sizeof(fenv_t)) return -1;
    fenv_t saved;
    memcpy(&saved,snapshot,sizeof(saved));
    /* Restoring tag metadata for live x87 stack DATA would be insufficient. */
    if (saved.__tags!=0xffffu) return -2;
    if ((saved.__control_word & 0x3fu)!=0x3fu ||
        (saved.__mxcsr & 0x1f80u)!=0x1f80u) return -3;
    if (saved.__mxcsr & 0xffff0000u) return -4;
    /* glibc fesetenv restores standard FE flags/modes but loses observed C2.
     * These instructions restore the actual saved 28-byte x87 environment and
     * saved MXCSR, retaining all 32 snapshot bytes, including condition bits.
     * No x87 arithmetic, stack pushes/pops or general register-context restore.
     */
    __asm__ volatile("fldenv %0\n\tldmxcsr %1"
        : : "m"(saved),"m"(saved.__mxcsr) : "memory");
    return 0;
}
