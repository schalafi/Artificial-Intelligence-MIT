from search import *
from graphs import *
from tester import make_test, get_tests
    
from lab2 import bfs, dfs, hill_climbing,beam_search,path_length,branch_and_bound,a_star 


def print_test(args,func, ans):
    """
    args: list
    func: Callable
    ans: list
    """
    val = func(*args)

    print("val: ",val)
    print("Expected: ", ans)
    print(val ==ans)
    print()

if __name__ == "__main__":
    print('Testing ...')

    from lab2 import hill_climbing

    print("Hill climbing: ")

    val = hill_climbing(NEWGRAPH1, 'S', 'G')
    correct = list('SABDCEG')
    print_test([NEWGRAPH1, 'S', 'G'], hill_climbing,correct )
    
   
    val = hill_climbing(NEWGRAPH1, 'F', 'G')
    correct = list('FBDCEG')
    print_test([NEWGRAPH1, 'F', 'G'], hill_climbing,correct )

    

    print("Beam search: ")
    val = beam_search(*[ NEWGRAPH1, 'S', 'G', 2 ])
    beam_search_1_answer = list('')
    print_test([ NEWGRAPH1, 'S', 'G', 2 ], beam_search,beam_search_1_answer)
    #print("path_length:")
    #print("cost: ", path_length(GRAPH2,['S','A','C','D'] ),26 )

    ### TEST 17 ###

    def beam_search_1_beam_10_getargs():
        return [ NEWGRAPH1, 'S', 'G', 10 ]

    beam_search_1_beam_10_answer = list('SCEG')

    def beam_search_1_beam_10_testanswer(val, original_val = None):
        return ( val == beam_search_1_beam_10_answer )

    print_test(beam_search_1_beam_10_getargs(), beam_search,beam_search_1_beam_10_answer )

    ### TEST 18 ###

    def beam_search_2_getargs():
        return [ NEWGRAPH1, 'F', 'G', 2 ]

    beam_search_2_answer = list('FBASCEG')
    def beam_search_2_testanswer(val, original_val = None):
        return ( val == beam_search_2_answer )

    print_test(beam_search_2_getargs(), beam_search,beam_search_2_answer )


    ### TEST 19 ###

    def beam_search_2_beam_10_getargs():
        return [ NEWGRAPH1, 'F', 'G', 10 ]
    beam_search_2_beam_10_answer = list('FBDEG')
    def beam_search_2_beam_10_testanswer(val, original_val = None):
        return ( val == beam_search_2_beam_10_answer )

    print_test(beam_search_2_beam_10_getargs(), beam_search,beam_search_2_beam_10_answer )


    ### TEST 20 ###
    def beam_search_3_beam_2_getargs():
        return [ NEWGRAPH2, 'S', 'G', 2 ]

    beam_search_3_beam_2_answer = list('')

    def beam_search_3_beam_2_testanswer(val, original_val = None):
        return ( val == beam_search_3_beam_2_answer )

    print_test(beam_search_3_beam_2_getargs(), beam_search,beam_search_3_beam_2_answer )


    ### TEST 21 ###

    def beam_search_3_beam_5_getargs():
        return [ NEWGRAPH2, 'S', 'G', 5 ]

    beam_search_3_beam_5_answer = list('SCDFG')

    def beam_search_3_beam_5_testanswer(val, original_val = None):
        return ( val == beam_search_3_beam_5_answer )

    print_test(beam_search_3_beam_5_getargs(), beam_search,beam_search_3_beam_5_answer )


    print("Brach and bound")
    def branch_and_bound_2_getargs():
        return [ NEWGRAPH1, 'S', 'D' ]

    def branch_and_bound_2_testanswer(val, original_val = None):
        return ( val == list('SCD') )


    print_test(branch_and_bound_2_getargs(),branch_and_bound, list('SCD'))

    def branch_and_bound_6_getargs():
        return [NEWGRAPH4, "S", "T"]

    def branch_and_bound_6_testanswer(val, original_val=None):
        return (val and list(val) == list("SBFHKT"))
    
    
    print_test(branch_and_bound_6_getargs(),branch_and_bound,list("SBFHKT"))


    print("A* ")
    ### TEST 25 ###

    def a_star_1_getargs():
        return [ NEWGRAPH3, 'S', 'S' ]

    def a_star_1_testanswer(val, original_val = None):
        return ( list(val) == list('S') )

    print_test(a_star_1_getargs(),a_star,list('S') )


    ### TEST 26 ###

    def a_star_2_getargs():
        return [ NEWGRAPH1, 'S', 'G' ]

    def a_star_2_testanswer(val, original_val = None):
        return ( list(val) == list('SCEG') )

    print_test(a_star_2_getargs(),a_star,list('SCEG') )


    ### TEST 27 ###

    def a_star_3_getargs():
        return [ NEWGRAPH2, 'S', 'G' ]

    def a_star_3_testanswer(val, original_val = None):
        return ( list(val) == list('SCDFG') )

    print_test(a_star_3_getargs(),a_star,list('SCDFG') )


    ### TEST 28 ###

    def a_star_4_getargs():
        return [ NEWGRAPH2, 'E', 'G' ]

    def a_star_4_testanswer(val, original_val = None):
        return ( list(val) == list('ECDFG') )

    print_test(a_star_4_getargs(),a_star,list('ECDFG') )

    ### TEST 30 ###

    def a_star_test_6_getargs():
        return [NEWGRAPH4, "S", "T"]

    def a_star_test_6_testanswer(val, original_val=None):
        return (list(val) == list("SBCJLT"))
    print_test(a_star_test_6_getargs(),a_star,list("SBCJLT") )

    ### TEST 31 ###

    def a_star_7_getargs():
        return [AGRAPH, "S", "G"]

    def a_star_7_testanswer(val, original_val=None):
        return (val and list(val) == list('SACG'))
                
    print_test(a_star_7_getargs(),a_star,list('SACG') )
    




