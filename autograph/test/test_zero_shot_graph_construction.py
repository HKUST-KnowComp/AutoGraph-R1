import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Set to the GPU you want to use
import asyncio
from  rag_server.reranker_api import Reranker
import time
import json_repair
from copy import deepcopy
title_lists = ["Radio City (Indian radio station)","History of Albanian football","Echosmith","Women's colleges in the Southern United States","First Arthur County Courthouse and Jail","Arthur's Magazine","2014\u201315 Ukrainian Hockey Championship","First for Women","Freeway Complex Fire","William Rast"]
nested_sentence_lists = [
    ["Radio City is India's first private FM radio station and was started on 3 July 2001.", " It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).", " It plays Hindi, English and regional songs.", " It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.", " Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.", " The Radio station currently plays a mix of Hindi and Regional music.", " Abraham Thomas is the CEO of the company."],
    ["Football in Albania existed before the Albanian Football Federation (FSHF) was created.", " This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .", " Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.", " In 1932, Albania joined FIFA (during the 12–16 June convention ) And in 1954 she was one of the founding members of UEFA."],
    ["Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.", " Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.", " Echosmith started first as \"Ready Set Go!\"", " until they signed to Warner Bros.", " Records in May 2012.", " They are best known for their hit song \"Cool Kids\", which reached number 13 on the \"Billboard\" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.", " The song was Warner Bros.", " Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.", " The band's debut album, \"Talking Dreams\", was released on October 8, 2013."],
    ["Women's colleges in the Southern United States refers to undergraduate, bachelor's degree–granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.", " Many started first as girls' seminaries or academies.", " Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.", " Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level."],
    ["The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum."],
    ["Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.", " Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.", " In May 1846 it was merged into \"Godey's Lady's Book\"."],
    ["The 2014–15 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.", " Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.", " Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.", " The regular season included just 12 rounds, where all the teams went to the semifinals.", " In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk."],
    ["First for Women is a woman's magazine published by Bauer Media Group in the USA.", " The magazine was started in 1989.", " It is based in Englewood Cliffs, New Jersey.", " In 2011 the circulation of the magazine was 1,310,696 copies."],
    ["The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.", " The fire started as two separate fires on November 15, 2008.", " The \"Freeway Fire\" started first shortly after 9am with the \"Landfill Fire\" igniting approximately 2 hours later.", " These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda."],
    ["William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.", " It is most known for their premium jeans.", " On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.", " The label also produces other clothing items such as jackets and tops.", " The company started first as a denim line, later evolving into a men’s and women’s clothing line."]
]
parsed_kg = [
    {"subject": "Radio City", "relation": "founded by", "object": "Abraham Thomas"},
    {"subject": "Radio City", "relation": "started on", "object": "3 July 2001"},
    {"subject": "Radio City", "relation": "broadcasts on", "object": "91.1 megahertz"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Mumbai"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Bengaluru"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Lucknow"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "New Delhi"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Hyderabad"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Chennai"},
    {"subject": "Radio City", "relation": "broadcasts from", "object": "Visakhapatnam"},
    {"subject": "Radio City", "relation": "plays", "object": "Hindi songs"},
    {"subject": "Radio City", "relation": "plays", "object": "English songs"},
    {"subject": "Radio City", "relation": "plays", "object": "regional songs"},
    {"subject": "Radio City", "relation": "launched", "object": "PlanetRadiocity.com"},
    {"subject": "Radio City", "relation": "launched", "object": "May 2008"},
    {"subject": "Albanian Football Federation (FSHF)", "relation": "created", "object": "1930"},
    {"subject": "Albania", "relation": "first international match against", "object": "Yugoslavia"},
    {"subject": "Albania", "relation": "first international match date", "object": "1946"},
    {"subject": "Albania", "relation": "joined FIFA", "object": "1932"},
    {"subject": "Albania", "relation": "joined UEFA", "object": "1954"},
    {"subject": "Echosmith", "relation": "formed in", "object": "February 2009"},
    {"subject": "Echosmith", "relation": "formed in", "object": "Chino, California"},
    {"subject": "Echosmith", "relation": "originally consisted of", "object": "four siblings"},
    {"subject": "Echosmith", "relation": "consists of", "object": "Sydney, Noah and Graham Sierota"},
    {"subject": "Echosmith", "relation": "started as", "object": "Ready Set Go!"},
    {"subject": "Echosmith", "relation": "signed to", "object": "Warner Bros. Records"},
    {"subject": "Echosmith", "relation": "signed to", "object": "May 2012"},
    {"subject": "Echosmith", "relation": "best known for", "object": "hit song 'Cool Kids'"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "reached number 13 on the Billboard Hot 100"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "certified double platinum by the RIAA"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "certified double platinum by ARIA in Australia"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "Warner Bros. Records' fifth-biggest-selling-digital song of 2014"},
    {"subject": "Echosmith", "relation": "hit song 'Cool Kids'", "object": "1.3 million downloads sold"},
    {"subject": "Echosmith", "relation": "debut album", "object": "Talking Dreams"},
    {"subject": "Echosmith", "relation": "debut album release date", "object": "October 8, 2013"},
    {"subject": "Salem College", "relation": "oldest female educational institution in the South", "object": "true"},
    {"subject": "Wesleyan College", "relation": "first college established specifically for women in the South", "object": "true"},
    {"subject": "First Arthur County Courthouse and Jail", "relation": "description", "object": "smallest courthouse in the United States"},
    {"subject": "First Arthur County Courthouse and Jail", "relation": "current use", "object": "museum"},
    {"subject": "Arthur's Magazine", "relation": "published by", "object": "T.S. Arthur"},
    {"subject": "Arthur's Magazine", "relation": "published in", "object": "Philadelphia"},
    {"subject": "Arthur's Magazine", "relation": "published during", "object": "1844–1846"},
    {"subject": "Arthur's Magazine", "relation": "merged into", "object": "Godey's Lady's Book"},
    {"subject": "Arthur's Magazine", "relation": "merged into", "object": "May 1846"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "season number", "object": "23"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "number of participating teams", "object": "4"},
    {"subject": "Generals Kiev", "relation": "participated in", "object": "2014–15 Ukrainian Hockey Championship"},
    {"subject": "Generals Kiev", "relation": "participated in", "object": "previous season"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "started after", "object": "year-end of 2014"},
    {"subject": "2014–15 Ukrainian Hockey Championship", "relation": "regular season rounds", "object": "12"},
    {"subject": "ATEK Kiev", "relation": "defeated", "object": "HK Kremenchuk"},
    {"subject": "ATEK Kiev", "relation": "defeated in", "object": "final"},
    {"subject": "First for Women", "relation": "started in", "object": "1989"},
    {"subject": "First for Women", "relation": "publisher", "object": "Bauer Media Group"},
    {"subject": "First for Women", "relation": "based in", "object": "Englewood Cliffs, New Jersey"},
    {"subject": "First for Women", "relation": "circulation", "object": "1,310,696 copies"},
    {"subject": "Freeway Complex Fire", "relation": "started on", "object": "November 15, 2008"},
    {"subject": "Freeway Complex Fire", "relation": "started first", "object": "Freeway Fire"},
    {"subject": "Freeway Complex Fire", "relation": "started first", "object": "9am"},
    {"subject": "Freeway Complex Fire", "relation": "started second", "object": "Landfill Fire"},
    {"subject": "Freeway Complex Fire", "relation": "started second", "object": "approximately 2 hours later"},
    {"subject": "William Rast", "relation": "founded by", "object": "Justin Timberlake and Trace Ayala"},
    {"subject": "William Rast", "relation": "started as", "object": "denim line"},
    {"subject": "William Rast", "relation": "evolved into", "object": "men’s and women’s clothing line"}
]

musique_docs =[
{
"idx": 0,
"title": "PolyGram Filmed Entertainment",
"paragraph_text": "PolyGram Filmed Entertainment (formerly known as PolyGram Films and PolyGram Pictures or simply PFE) was a British-American film studio founded in 1980 which became a European competitor to Hollywood, but was eventually sold to Seagram Company Ltd. in 1998 and was folded in 1999. Among its most successful and well known films were \"An American Werewolf in London\" (1981), \"Flashdance\" (1983), \"Four Weddings and a Funeral\" (1994), \"Dead Man Walking\" (1995), \"The Big Lebowski\" (1998), \"Fargo\" (1996), \"The Usual Suspects\" (1995), and \"Notting Hill\" (1999).",

},
{
"idx": 1,
"title": "Fortnite",
"paragraph_text": "Fortnite is a co-op sandbox survival game developed by Epic Games and People Can Fly and published by Epic Games. The game was released as a paid - for early access title for Microsoft Windows, macOS, PlayStation 4 and Xbox One on July 25, 2017, with a full free - to - play release expected in 2018. The retail versions of the game were published by Gearbox Publishing, while online distribution of the PC versions is handled by Epic's launcher.",

},
{
"idx": 2,
"title": "6 Nimmt!",
"paragraph_text": "6 Nimmt! / Take 6! is a card game for 2-10 players designed by Wolfgang Kramer in 1994 and published by Amigo Spiele. The French version is distributed by Gigamic. This game received the \"Deutscher Spiele Preis\" award in 1994.",

},
{
"idx": 3,
"title": "The Price Is Right",
"paragraph_text": "Endless Games, which in the past has produced board games based on several other game shows, including The Newlywed Game and Million Dollar Password, distributes home versions of The Price Is Right, featuring the voice of Rich Fields, including a DVD edition and a Quick Picks travel - size edition. Ubisoft also released a video game version of the show for the PC, Nintendo DS, and Wii console on September 9, 2008. An updated version of the game (The Price Is Right: 2010 Edition) was released on September 22, 2009. Both versions feature the voice of Rich Fields, who was the show's announcer at the time of the release of the video games in question.",

},
{
"idx": 4,
"title": "Hinterland (video game)",
"paragraph_text": "Hinterland is a high fantasy role-playing video game with city-building elements by Tilted Mill Entertainment. It was released on September 30, 2008 on the Steam content delivery system, and has since been made available at other digital distribution websites. Hinterland: Orc Lords, a cumulative expansion, was released to digital distribution and retail in March 2009. As the title suggests, the primary addition to the game was the ability to play as Orc characters.",

},
{
"idx": 5,
"title": "2018 Asian Games medal table",
"paragraph_text": "The 2018 Asian Games, officially known as the XVIII Asiad, is the largest sporting event in Asia governed by Olympic Council of Asia (OCA). It was held at Jakarta and Palembang, Indonesia between 18 August -- 2 September 2018, with 465 events in 40 sports and disciplines featured in the Games. This resulted in 465 medal sets being distributed.",

},
{
"idx": 6,
"title": "Nemrem",
"paragraph_text": "Nemrem, known as Zengage in North America and Somnium in Japan, is a puzzle video game developed by Skip Ltd. and published by Nintendo for the Nintendo DSi's DSiWare digital distribution service.",

},
{
"idx": 7,
"title": "Day of Defeat: Source",
"paragraph_text": "Day of Defeat: Source is a team-based online first-person shooter multiplayer video game developed by Valve Corporation. Set in World War II, the game is a remake of \"Day of Defeat\". It was updated from the GoldSrc engine used by its predecessor to the Source engine, and a remake of the game models. The game was released for Microsoft Windows on September 26, 2005, distributed through Valve's online content delivery service Steam. Retail distribution of the game was handled by Electronic Arts.",

},
{
"idx": 8,
"title": "Nintendo Entertainment System",
"paragraph_text": "The Nintendo Entertainment System (commonly abbreviated as NES) is an 8 - bit home video game console that was developed and manufactured by Nintendo. It was initially released in Japan as the Family Computer (Japanese: ファミリーコンピュータ, Hepburn: Famirī Konpyūta) (also known by the portmanteau abbreviation Famicom (ファミコン, Famikon) and abbreviated as FC) on July 15, 1983, and was later released in North America during 1985, in Europe during 1986 and 1987, and Australia in 1987. In South Korea, it was known as the Hyundai Comboy (현대컴보이 Hyeondae Keomboi) and was distributed by SK Hynix which then was known as Hyundai Electronics. The best - selling gaming console of its time, the NES helped revitalize the US video game industry following the video game crash of 1983. With the NES, Nintendo introduced a now - standard business model of licensing third - party developers, authorizing them to produce and distribute titles for Nintendo's platform. It was succeeded by the Super Nintendo Entertainment System.",

},
{
"idx": 9,
"title": "Nintendo",
"paragraph_text": "Nintendo's first venture into the video gaming industry was securing rights to distribute the Magnavox Odyssey video game console in Japan in 1974. Nintendo began to produce its own hardware in 1977, with the Color TV - Game home video game consoles. Four versions of these consoles were produced, each including variations of a single game (for example, Color TV Game 6 featured six versions of Light Tennis).",

},
{
"idx": 10,
"title": "Kakuto Chojin: Back Alley Brutal",
"paragraph_text": "Kakuto Chojin: Back Alley Brutal (Kakuto Chojin for short), known in Japan as , is a fighting game for the Xbox gaming console published in 2002 by Microsoft Game Studios. The game was the sole product of developer Dream Publishing, a studio created from members of Dream Factory and Microsoft. It was originally created as a tech demo to show off the graphic capabilities of the Xbox, before the decision was made to turn it into a full game. A few months after its release, \"Kakuto Chojin\" was pulled from distribution amidst controversy surrounding the religious content featured in the game.",

},
{
"idx": 11,
"title": "Third quarterback rule",
"paragraph_text": "The third quarterback rule was a rule in the National Football League that governed the use of a third quarterback in addition to the starter and the backup. The rule was abolished for the 2011 season, when the NFL increased the roster size to allow 46 players to dress for a game.",

},
{
"idx": 12,
"title": "The Ball Game",
"paragraph_text": "The Ball Game is an 1898 American short black-and-white silent documentary sports film produced and distributed by Edison Manufacturing Company.",

},
{
"idx": 13,
"title": "Flock!",
"paragraph_text": "Flock! (stylized as FLOCK!) is a puzzle video game developed by Proper Games and published by Capcom for Windows, PlayStation Network and Xbox Live Arcade. It was released for Microsoft Windows on April 7, 2009 through Steam and Stardock's digital distribution service Impulse, Xbox Live Arcade on April 8, 2009 and PlayStation Network on April 9, 2009.",

},
{
"idx": 14,
"title": "Pokémon Conquest",
"paragraph_text": "Pokémon Conquest, known in Japan as , is a tactical role-playing video game developed by Tecmo Koei, published by The Pokémon Company and distributed by Nintendo for the Nintendo DS. The game is a crossover between the \"Pokémon\" and \"Nobunaga's Ambition\" video game series. The game was released in Japan on March 17, 2012, in North America on June 18, 2012, and in Europe on July 27, 2012.",

},
{
"idx": 15,
"title": "Smashing Drive",
"paragraph_text": "Smashing Drive is a racing video game developed and published by Gaelco and distributed by Namco. The game was first released in arcades in 2000 and was ported to the Nintendo Gamecube and Xbox in 2002 by Point of View and Namco. Subsequently, it has been brought to the Game Boy Advance in 2004 by DSI Games and Namco.",

},
{
"idx": 16,
"title": "MaxPlay Classic Games Volume 1",
"paragraph_text": "MaxPlay Classic Games Volume 1 is a compilation of video games developed by CodeJunkies and published by parent company Datel for the Nintendo GameCube and PlayStation 2. The title was distributed by Ardistel in Europe; it was included as a gift with the purchase of Datel's \"Max Memory\" memory card for the PlayStation 2. It is the only unlicensed game for the GameCube and one of only a handful for the PlayStation 2. The collection features ten games, all unofficial remakes of classic games. It includes:",

},
{
"idx": 17,
"title": "The Game (1997 film)",
"paragraph_text": "The Game is a 1997 American mystery thriller film directed by David Fincher, starring Michael Douglas and Sean Penn, and produced by Propaganda Films and PolyGram Filmed Entertainment. It tells the story of a wealthy investment banker who is given a mysterious gift: participation in a game that integrates in strange ways with his everyday life. As the lines between the banker's real life and the game become more uncertain, hints of a large conspiracy become apparent.",

},
{
"idx": 18,
"title": "Pokémon Diamond and Pearl",
"paragraph_text": "Pokémon Diamond Version and Pearl Version (ポケットモンスターダイヤモンド・パール, Poketto Monsutā Daiyamondo & Pāru, ``Pocket Monsters: Diamond & Pearl '') are role - playing games (RPGs) developed by Game Freak, published by The Pokémon Company and distributed by Nintendo for the Nintendo DS. With the enhanced remake Pokémon Platinum, the games comprise the fifth installment and fourth generation of the Pokémon series of RPGs. First released in Japan on September 28, 2006, the games were later released to North America, Australia, and Europe over the course of 2007.",

},
{
"idx": 19,
"title": "Super Mario Bros. (film)",
"paragraph_text": "Super Mario Bros. is a 1993 American fantasy adventure film based on the Japanese video game series of the same name and the game Super Mario Bros. by Nintendo. It was directed by Rocky Morton and Annabel Jankel, written by Parker Bennett, Terry Runté and Ed Solomon, and distributed by Walt Disney Studios through Hollywood Pictures.",

}
]
musique_text = [f'Text Title:{doc["title"]}, text:{doc["paragraph_text"]}' for doc in musique_docs]
# nested_sentence_lists = nested_sentence_lists[:1]
# title_lists = title_lists[:1]

prompt = (
    """You are an expert knowledge graph constructor. Your task is to extract factual information from the provided text and represent it as a list of knowledge graph triples.
    Each triple should be a JSON object with three keys:
    1.  `subject`: The main entity, concept, event, or attribute of the triple.
    2.  `relation`: The relationship between the subject and the object.
    3.  `object`: The entity, concept, value, event, or attribute that the subject has a relationship with.
    Constraints:
    - Extract all possible and relevant triples.
    - The `subject` and `object` can be specific entities (e.g., "Radio City", "Football in Albania", "Echosmith") or specific values (e.g., "3 July 2001", "1,310,696").
    - The `relation` should be a concise, descriptive phrase or verb that accurately describes the relationship (e.g., "founded by", "started on", "is a", "has circulation of").
    - Ensure the triples are self-contained and logically sound.
    - If no triples can be extracted from the text, return an empty JSON list: `[]`.
    - Do not include any text other than the JSON output."""
)
prompt_raw_text = []
for title, doc in zip(title_lists, nested_sentence_lists):
    passage = "".join(doc)
    prompt_raw_text.append(f"{title}: {passage}")

import configparser
from rag_server.llm_api import LLMGenerator
from openai import AsyncOpenAI, OpenAI
import numpy as np
config_parser = configparser.ConfigParser()
config_parser.read("../verl/third_party/autograph_r1/config.ini")
config = config_parser['vllm']
api_url = config['URL']
api_key = config['KEY']
print(f"Using API URL: {api_url} and API Key: {api_key}")
# client = AsyncOpenAI(
#     base_url=api_url,
#     api_key=api_key,
#     timeout=300
# )
if __name__ == "__main__":

    loop = asyncio.get_event_loop()

    loop_generate = True
    test_reranker = False
    start_time = time.time()
    if test_reranker:
        config = config_parser['vllm_emb']
        api_url = config['URL']
        api_key = config['KEY']
        emb_client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )
        reranker = Reranker(
            emb_client
        )
        input_texts = ["China","Bejing"]
        docs = ["Bejing", "Shanghai", "Texas"]
        sim_scores = reranker.compute_similarity(["China", "USA"], docs)
        index = np.argmax(sim_scores, axis=1)
        most_similar_nodes = [docs[i] for i in index]
        print(f"Similarity scores: {sim_scores}")
        print(f"Most similar nodes: {most_similar_nodes}")

    else:
        llm_engine = AsyncOpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=300
        )

        llm_api = LLMGenerator(
            client=llm_engine,
            # model_name='Qwen/Qwen2.5-14B-Instruct'
            model_name = 'Qwen/Qwen2.5-7B-Instruct',
            backend='vllm'
        )
        if loop_generate:
            target_messages_list = [
                {
                    "role": "system", # 0
                    "content": f"{prompt}"
                },
            ]
            triples = []
            num_tokens= 0
            for doc in musique_text:
                prompt = deepcopy(target_messages_list)
                prompt.append(
                    {
                        "role": "user", # 1
                        "content": f"{doc}"
                    }
                )
                response = loop.run_until_complete(llm_api.generate_response(prompt, temperature=0, return_text_only = True, max_new_tokens=24576))
                print(f"Response: {response}")
                num_tokens = len(response.split())
                triples.extend(json_repair.loads(response))
            print(f'Number of triples: {len(triples)}')
            print(f'Number of tokens: {num_tokens}')

        else:
            num_tokens = 0
            docs = "\n".join(musique_text)
            target_messages_list = [
                {
                    "role": "system", # 0
                    "content": f"{prompt}"
                },
                {
                    "role": "user", # 1
                    "content": f"{docs}"
                    # "content": f"Elon musk owns Twitter, which was started on 2006. Elon Musk is the CEO of Twitter. Twitter is a social media platform."
                },
            ]
            response = loop.run_until_complete(llm_api.generate_response(target_messages_list, temperature=0, return_text_only = True, max_new_tokens=24576))
            print(f"Response: {response}")
            num_tokens = len(response.split())
            print(f'Number of tokens: {num_tokens}')
            triples = json_repair.loads(response)
            print(f'Number of triples: {len(triples)}')
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
