hir.func.extern @mult_3stage at %t (%a:i32,%b:i32)->(%result:i32 delay 2)
{argNames=["a","b","t"],resultNames=["result"]}
hir.func @mac at %t (%a :i32, %b :i32, %c :i32) -> (%result: i32 delay 3){
  %m = hir.call "mult" @mult_3stage (%a,%b) at %t 
  : !hir.func<(i32, i32) -> (i32 delay 2)>
  %c2= hir.delay %c by 2 at %t : i32 
  %res = comb.add  %m,%c2  : i32
  %res_delayed = hir.delay %res by 1 at %t + 2: i32
  hir.return (%res_delayed) : (i32)
}{argNames=["a","b","c","t"],resultNames=["result"]}