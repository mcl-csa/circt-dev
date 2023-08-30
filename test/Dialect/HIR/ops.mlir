hir.func.extern @mul at %t(%a:i32,%b:i32) -> (%c:i32 delay 1){argNames=["a","b","t"],resultNames=["c"]}

hir.func @test1 at %t (%a: i32, %b: i32, %c: i32 delay 1) {
    %f = hir.instance @mul :!hir.func<(i32,i32)->(i32 delay 1)>
    %m = hir.call_instance %f(%a,%b) at %t :!hir.func<(i32,i32)->(i32 delay 1)>
    %r = hir.call_instance %f(%m,%c) at %t+1 :!hir.func<(i32,i32)->(i32 delay 1)>
}{argNames=["a","b","c","t"]}
