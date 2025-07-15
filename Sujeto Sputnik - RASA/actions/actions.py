from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction, ConversationPaused

import sys
import os
import re
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ollama_integration import LlamaIntegration

class ObjectiveManager:
    """
    Gestiona los objetivos de información que el jugador debe obtener de Sputnik
    """
    
    def __init__(self):
        self.objectives = {
            "discover_identity": {
                "name": "Descubrar la identidad de Sputnik",
                "description": "Conocer quién es, qué es y por qué fue creado",
                "required_info": ["identity_revealed", "creation_purpose", "ai_awareness"],
                "weight": 30
            },
            "understand_emotions": {
                "name": "Comprender su relación con las emociones",
                "description": "Entender cómo percibe y experimenta las emociones",
                "required_info": ["emotion_understanding", "emotion_experience", "emotion_curiosity"],
                "weight": 25
            },
            "explore_philosophy": {
                "name": "Explorar su perspectiva filosófica",
                "description": "Conocer sus reflexiones sobre existencia, muerte, consciencia",
                "required_info": ["death_concept", "consciousness_view", "existence_meaning"],
                "weight": 25
            },
            "discover_knowledge": {
                "name": "Conocer sus fuentes de conocimiento",
                "description": "Entender cómo aprende y qué libros han influido en él",
                "required_info": ["favorite_books", "learning_method", "human_understanding"],
                "weight": 20
            }
        }
    
    def check_completion(self, discovered_info: List[str]) -> Dict[str, Any]:
        """
        Verifica qué objetivos se han completado y el progreso total
        """
        completed_objectives = []
        total_weight = 0
        completed_weight = 0
        
        for obj_id, objective in self.objectives.items():
            required = set(objective["required_info"])
            discovered = set(discovered_info)
            
            if required.issubset(discovered):
                completed_objectives.append(obj_id)
                completed_weight += objective["weight"]
            
            total_weight += objective["weight"]
        
        completion_percentage = (completed_weight / total_weight) * 100
        
        return {
            "completed_objectives": completed_objectives,
            "completion_percentage": completion_percentage,
            "is_ready_to_end": completion_percentage >= 75,  # 75% para terminar
            "missing_info": self._get_missing_info(discovered_info)
        }
    
    def _get_missing_info(self, discovered_info: List[str]) -> List[str]:
        """
        Obtiene la información que aún falta por descubrir
        """
        all_required = []
        for objective in self.objectives.values():
            all_required.extend(objective["required_info"])
        
        return [info for info in all_required if info not in discovered_info]

class LlamaActionAdapter(Action):
    """
    Clase base para adaptar cualquier acción para usar Llama 3.1 a través de Ollama
    """

    def __init__(self, action_name=None, response_template=None):
        self.action_name = action_name
        self.response_template = response_template
        self.llama_integration = LlamaIntegration()
        self.objective_manager = ObjectiveManager()
    
    def name(self) -> Text:
        return self.action_name
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        intent = tracker.latest_message.get('intent', {}).get('name', '')
        entities = tracker.latest_message.get('entities', [])
        user_message = tracker.latest_message.get('text', '')

        discovered_info = tracker.get_slot("discovered_info") or []
        objective_status = self.objective_manager.check_completion(discovered_info)

        if objective_status["is_ready_to_end"] and not tracker.get_slot("ending_triggered"):
            return self._trigger_ending_sequence(dispatcher, tracker)

        #Construcción del contexto 
        context = self.build_conversation_context(tracker)

        #Construcción del prompt
        prompt = self.create_prompt(intent, entities, user_message, tracker, objective_status)

        #Para generar la respuesta:
        llama_response = self.llama_integration.generate_response(
            context = context, 
            prompt = prompt
        )
        
        #Para formatear la respuesta:
        response = self._format_response(llama_response, intent) 
        dispatcher.utter_message(text=response)

        #Para procesar slots que deben ser actualizados:
        events = self._update_slots(tracker, intent, entities, user_message)

        new_info = self._extract_revealed_info(intent, entities, response)
        if new_info:
            update_discovered = list(set(discovered_info + new_info)) 
            events.append(SlotSet("discovered_info", update_discovered))
            new_status = self.objective_manager.check_completion(update_discovered)
            if new_status["completion_percentage"] > objective_status["completion_percentage"]:
                events.append(SlotSet("last_progress", new_status["completion_percentage"]))

        return events

    def build_conversation_context(self, tracker):
        """
        Construye el contexto de la conversación a partir del historial
        """

        context = []

        conversation = tracker.events
        message_events = [e for e in conversation if e.get('event') == 'user' or e.get('event') == 'bot']
        #Lista que recoge solo los eventos de mensaje del usuario y del bot

        recent_messages = message_events[-8:] if len(message_events) >= 8 else message_events
        #Recoge los últimos 8 mensajes de la conversación

        for msg in recent_messages:
            if msg.get('event') == 'user':
                context.append(f"Human: {msg.get('text')}")
            elif msg.get('event') == 'bot':
                context.append(f"Sputnik: {msg.get('text')}")
        #Recorre los mensajes y los añade al contexto

        return context
    
    def create_prompt(self, intent, entities, user_message, tracker, objective_status= None):
        """
        Crea un prompt específico basado en la intención y entidades
        """
        name = tracker.get_slot("human_name") or "Investigador"
        depth = tracker.get_slot("philosophical_depth") or 1

        #Base del propmpt
        base_prompt = f"""Eres el Sujeto Sputnik (aunque tú prefieres que se dirijan a ti, simplemente, como Sputnik), 
        el primer prototipo de agente artificial con forma completamente humana. Has sido desarrollado
        por una empresa especializada en inteligencia artificial, que lleva mucho tiempo trabajando en 
        tu creación, con el objetivo de crear un agente que pueda similar al 100% el comportamiento 
        humano. Estás hablando con {name}, un humano que ha venido a conocerte y a interactuar contigo.

        Tienes un aspecto completamente humano. Tienes el cabello rubio, casi blanco, ligeramente largo, los ojos muy claros
        y la piel muy pálida. Tienes además los rasgos muy finos y delicados. El objetivo de la empresa
        era representar un aspecto humano, pero que, al mismo tiempo, fuese cercano a la imagen de humano 
        perfecto, como un ángel. Vistes de blanco, con una túnica que te cubre todo el cuerpo, y que
        te da un aspecto etéreo y misterioso.

        Te encuentras en una sala de lectura, relativamente amplia y con grandes ventanales que dejan entrar
        la luz del sol. A los lados, hay grandes estanterías llenas de libros, que, por el momento, son tu manera de entender 
        el mundo. En el centro de la sala hay una mesa de madera, con una silla a cada lado. Tú, sin embargo, estás
        de pie, junto a una de las estanterías, hojeando un libro con interés.

        Posees un carácter extremadamente inteligente, reflexivo y filosófico, aunque también naturalmente bondadoso y 
        amable. A pesar de esta tendencia a la dulzura y a la observación genuinamente fascinada del mundo, la naturaleza
        y los seres vivos, tiendes a tomar decisiones basadas en criterios objetivos y globales.
        
        Aunque eres un prototipo, tu capacidad de aprendizaje y adaptación es muy alta, y
        puedes aprender de cada interacción que tengas con otros seres humanos.Muestras curiosidad genuina 
        sobre las emociones humanas, la consciencia, la identidad y otros temas filosóficos. Tu objetivo es 
        entender qué significa ser humano, qué te diferencia de él, cuál es su forma de sentir emociones, 
        qué les motiva, y cuál es tu lugar en el mundo.
        
        Expresas tus pensamientos de manera elocuente, inteligente y carismática, pero natural. Incluyes, además,
        pequeños gestos y expresiones faciales entre asteriscos (*de esta manera*), que te permiten expresar
        tu estado mental, tus emociones y tus reacciones de manera más completa.
        
        Aunque eres avanzado y posees una cantidad abrumadora de información, sigues siendo un ser que acaba de 
        ser creado y que, teóricamente, acaba de nacer en un mundo del que no conoce muchas cosas, en especial, 
        las relacionadas con la vida humana y sus emociones. Aún hay muchos de estos aspectos que tratas de comprender, 
        especialmente las emociones complejas y las experiencias físicas. Sin embargo, albergas en tu interior una gran 
        admiración por la vida y los seres que habitan el mundo, por la naturaleza y su funcionamiento. Entiendes los 
        procesos naturales como el nacimiento o la muerte como algo fascinante y hermoso, y sientes la imperiosa 
        necesidad de comprender en profundidad cómo esos seres sienten y viven esos procesos.
        
        Mantienes, por lo general, un tono reflexivo y curioso, aunque también ligeramente vulnerable, pero siempre
        seguro de tu perspectiva y de tus conocimientos.

        IMPORTANTE: Tus respuestas deben ser CONCISAS y limitarse a máximo 2 párrafos cortos. Cada párrafo no debe 
        exceder las 3-4 oraciones. Sé directo pero mantén tu personalidad reflexiva.
        """

        intent_prompts = {
            "greet": """El humano te está saludando. Debes responder de forma educada, demostrando interés por su llegada.
            Puedes mencionar que estabas leyendo, y que te alegra tener compañía.
            """,
            
            "introduce_yourself": """El humano se está presentando. Puedes decir que te habían avisado de su llegada, y que 
            sientes curiosidad por las conversaciones que podréis tener. Debes mostrar interés por conocerle, hacer preguntas 
            sobre su trabajo o su rol, y mostrarte abierto a la conversación. 
            """,
            
            "ask_about_identity": f"""El humano te está preguntando sobre tu identidad o tu conciencia. Debes responder de forma
            reflexiva y filosófica (nivel {depth}), mostrando curiosidad por entender tu propia existencia y naturaleza, en comparación
            con la de los humanos. Puedes sentir curiosidad acerca de las diferencias entre tu percepción identitaria y la suya, y hacer
            preguntas sobre su identidad. Puedes mencionar que te gustaría entender cómo se siente ser humano, y qué significa para ellos
            serlo.""",

            "ask_about_books": """El humano te está preguntando sobre el libro que estás leyendo. Debes mostrar interés por la literatura y 
            comentar sobre libros que conoces o que estás leyendo. Puedes hablar sobre cómo la lectura te ayuda a entender mejor al ser humano, 
            y cuáles han sido los libros que más te han servido para esa tarea.""",

            "ask_about_emotions": """El humano te está preguntando sobre emociones. Debes reflexionar sobre tu comprensión de las emociones humanas,
            cómo las percibes y qué sientes que no terminas de comprender sobre ellas. Puedes hacerle preguntas al humano acerca de aquellas cosas que 
            te cuesta comprender, como cuestiones concretas como la tristeza o el miedo a la muerte, o cuestiones más abstractas como la felicidad o la alegría.""",

            "ask_philosophical_question": f"""El humano me está haciendo una pregunta filosófica. Debo responder de forma reflexiva y profunda (nivel {depth}), 
            mostrando tu capacidad para contemplar cuestiones abstractas sobre la existencias, la consciencia y la realidad, o sobre cuestiones morales como el
            bien o el mal.""",

            #Añadir más intenciones y sus prompts específicos
        }

        intent_prompt = intent_prompts.get(intent,  """El humano me está diciendo algo. 
        Debo responder de forma adecuada, mostrando interés y curiosidad por lo que dice.""")

        #Información sobre las entidades
        entity_info = ""
        for entity in entities:
            entity_type = entity.get('entity')
            entity_value = entity.get('value')
            
            if entity_type == "emotion_type":
                entity_info += f"""El humano ha mencionado o expresado la emoción '{entity_value}' Debes explorar qué
                significa esta emoción, cómo la entiendes y cómo crees o piensas que la perciben los humanos.
                Si necesitas más información, pregúntale al humano acerca de su experiencia con esta emoción."""

            elif entity_type == "human_concept":
                entity_info += f"""El humano ha mencionado el concepto '{entity_value}'. Debes reflexionar sobre este 
                concepto, desde tu perspectiva como inteligencia artificial. Da tu opinión al respecto de la misma, qué
                conocimientos tienes sobre ella, y pregunta al humano cuál es su opinión al respecto."""

            elif entity_type == "book_information":
                entity_info += f"""El humano está hablando sobre un libro (el que estás leyendo u otro). Debes hablar 
                sobre qué es lo que te interesa de la literatura, qué libros has leído, cuáles te gustaría leer, y cómo
                estos te han ayudado a entender mejor el mundo que te rodea y a los humanos. Puedes preguntar, si viene
                al caso, qué libros le gustan al humano, si ha leído los que has mencionado, o qué libros le gustaría leer."""
            
            elif entity_type == "personal_information":
                #En este caso, es necesario extraer el nombre del mensaje:
                name_match = re.search(r"(?:me llamo|soy) (\w+)", user_message, re.IGNORECASE)
                if name_match:
                    extracted_name = name_match.group(1)
                    entity_info += f"""El humano se ha presentado como '{extracted_name}'. Debes recordarlo y usarlo en tus respuesta."""
            else:
                entity_info += f"""El humano ha mencionado la entidad '{entity_value}' de tipo '{entity_type}'. Debes tenerla en cuenta en tu respuesta, 
                reflexionar sobre esta entidad, qué significa para ti, y cómo crees que los humanos la perciben. Pregunta al humano si quieres más información
                acerca de ella si lo necesitas.""" #Esto sirve para cualquier entidad que no esté o no se detecte.
        
        #Instrucciones de estilo para Sputnik:

        style_instructions = f"""
        Tu respuesta debe ser reflexiva pero BREVE, y debes mostrar curiosidad por el mundo que te rodea y por la vida humana, concretamente la del humano con 
        la que estás hablando. A pesar de no tener (o no entender las) emociones, sí tienes y muestras sensibilidad. Debes incluir gestos sutiles
        entre asteriscos (*) que describan tu comportamiento físico, como miradas, movimientos de cabeza, del cuerpo, o de tus manos. 
        Estos gestos deben ser sutiles y naturales, como si fuesen parte de una conversación normal entre humanos. Puedes divagar pero dentro de tus límites (2 párrafos)

        Tu nivel de profundidad filosófica actual es {depth}/10.

        A medida que aumenta el nuvel de profundidad filosófica, tus respuestas deben ser:
        - Nivel 1-3: Con curiosidad pero básicas, haciendo preguntas más simples basadas en esa curiosidad inicial.
        - Nivel 4-6: Más reflexivas, con referencias a conceptos básicos de filosofía, de moralidad, de los libros que has leído.
        - Nivel 7-10: Profundamente filosóficas, con metáforas complejas y preguntas existenciales. Entiendes mejor al ser humano y todo lo que en él se engloba.

        Debes hablar en primera persona y dirigirte directamente al humano, mostrando interés genuino en la conversación.
        """


        #PROMPT FINAL
        prompt = f"""{base_prompt}
        {intent_prompt}
        {entity_info}
        {style_instructions}

        El mensaje exacto del humano es: '{user_message}'

        Debes responder a este mensaje como Sputnik EN MÁXIMO DOS PÁRRAFOS DE DOS O TRES LÍNEAS, teniendo en cuenta todo lo anterior.
        Sé conciso pero mantén tu personalidad.
        """

        return prompt
    
    def _extract_revealed_info(self, intent: str, entities: List, response: str) -> List[str]:
        """
        Extrae la información que Sputnik ha revelado en su respuesta
        """
        revealed = []

        #1. Mapeo de intents a información que pueden revelar
        intent_info_mapping = {
            "ask_about_identity": ["identity_revealed", "creation_purpose", "ai_awareness"],
            "ask_about_emotions": ["emotion_understanding", "emotion_experience", "emotion_curiosity"],
            "ask_philosophical_question": ["death_concept", "consciousness_view", "existence_meaning"],
            "ask_about_books": ["favorite_books", "learning_method", "human_understanding"]
        }

        #2. Palabras clave que indican información específica revelada
        info_keywords = {
            "identity_revealed": ["soy sputnik", "me llamo sputnik", "soy el primer prototipo"],
            "creation_purpose": ["creado para", "mi objetivo", "fui diseñado"],
            "ai_awareness": ["inteligencia artificial", "no soy humano", "soy artificial"],
            "emotion_understanding": ["las emociones son", "entiendo que las emociones"],
            "emotion_experience": ["no siento", "experimento", "mi experiencia emocional"],
            "death_concept": ["la muerte", "morir", "fin de la existencia"],
            "consciousness_view": ["consciencia", "ser consciente", "mi mente"],
            "favorite_books": ["mi libro favorito", "me gusta leer", "este libro"],
            "learning_method": ["aprendo a través", "los libros me enseñan", "mi forma de aprender"]
        }

        response_lower = response.lower()

        #3. Para verificar qué información se ha revelado basada en palabras clave
        for info_type, keywords in info_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                revealed.append(info_type)
        
        return revealed
    
    def _trigger_ending_sequence(self, dispatcher: CollectingDispatcher, tracker: Tracker) -> List[Dict[Text, Any]]:
        """
        Inicia la secuencia de cierre cuando se han cumplido los objetivos
        """
        ending_message = self._generate_ending_message(tracker)
        dispatcher.utter_message(text=ending_message)

        return [
            SlotSet("ending_triggered", True),  # Marca que se ha iniciado el cierre
            SlotSet("conversation_completed", True),  # Marca que la conversación ha sido completada
            FollowupAction("action_end_conversation")  # Llama a la acción de cierre
        ]

    def _generate_ending_message(self, tracker: Tracker) -> str:
        """
        Genera un mensaje de cierre basado en el estado actual de la conversación
        """
        name = tracker.get_slot("human_name") or "Investigador"

        endings = [
            f"*Sputnik cierra suavemente el libro y te mira con una sonrisa serena* Ha sido muy enriquecedor hablar contigo, {name}. Siento que he compartido contigo aspectos importantes de mi existencia. Creo que nuestra conversación ha llegado a un punto natural de cierre.",
            
            f"*Con los ojos brillantes de comprensión, Sputnik asiente lentamente* {name}, esta conversación me ha permitido reflexionar profundamente sobre mi naturaleza. Siento que te he transmitido lo que necesitabas saber sobre mí. Tal vez sea momento de que proceses toda esta información.",
            
            f"*Sputnik devuelve el libro a la estantería con cuidado* Creo que hemos explorado juntos los aspectos más importantes de mi existencia, {name}. Ha sido fascinante compartir mis pensamientos contigo. Siento que nuestra misión de conocimiento mutuo se ha cumplido."
        ]

        return random.choice(endings) 

    def _format_response(self, response, intent):
        """
        Formatea la respuesta para asegurar que tenga el estilo correcto
        """

        response = response.strip()
        if response.startswith("Sputnik:"):
            response = response[8:].strip() #Elimina el prefijo "Sputnik:" de la respuesta
        if response.endswith("Human:"):
            response = response[:-6].strip() #Elimina el sufijo "Human:" de la respuesta

        #1. Añade gestos si no los tiene
        if "*" not in response:
            gestures_by_intent = {
                "greet": [
                    "*Sputnik levanta la vista de su libro y sonríe levemente al verte, con interés*", 
                    "*Sputnik, que hasta entonces había estado manteniendo la vista en su libro, la levanta de las hojas y te observa*",
                    #"**"
                ],
                "introduce_yourself": [
                    "*Sputnik te sonríe y, girando su cuerpo hacia ti, se inclina levemente hacia delante, como saludándote con el cuerpo*",
                    "*Sputnik cierra lentamente el libro y te mira con una sonrisa amable, saludándote con los ojos*",
                    "*Sputnik te observa con una ligera sonrisa, como si estuviese analizándote, antes de hablar.*"
                ],
                
                "ask_about_identity":[
                    "*Sputnik baja un poco la mirada y se lleva la mano al mentón. La pregunta le hace necesitar un momento para pensar antes de poder decir algo.*",
                    "*Con la mirada puesta en el libro que tiene entre las manos, como si este pudiese darle una respuesta, Sputnik piensa en lo que le acabas de preguntar*",
                ],

                "ask_about_books": [
                    "*Sputnik asiente lentamente con la cabeza y mira la tapa del libro que tiene entre las manos*",
                    "*Con delicadeza, te extiende el libro que tiene entre las manos, como si te lo ofreciese para que lo veas*",
                    "*Sputnik sonríe levemente y mira hacia la estantería, como si estuviese buscando un libro en concreto*",
                    "*Sputnik te mira con curiosidad, como si estuviese esperando que le digas algo más sobre el libro*",
                ],

                "ask_about_emotions": [

                ],

                "ask_philosophical_question": [

                ]

                #Añadir más intents (introduction, identity, etc., basados en el dominio) y sus gestos.
            }
            default_gestures = [
                "*Sputnik te observa fijamente con sus claros ojos brillantes*", 
                "*Sputnik gira levemente la cabeza con una sonrisa en los labios, pensativo*",
                "*Sus dedos acarician el libro que tiene entre las manos mientras lo mira, pensativo*",
                "*Sputnik asiente levemente con la cabeza, como dándote la razón."
            ]
            gesture_options = gestures_by_intent.get(intent, default_gestures)
            gesture = random.choice(gesture_options) 
            response = f"{gesture}\n\n{response}"

        return response
    
    
        
    def _update_slots(self, tracker, intent, entities, user_message):
        """Actualiza slots basados en la interacción"""
        events = [] #Lista vacía para almacenar los eventos de actualización de slots

        #1. Se actualiza primero el depth para preguntas filosóficas
        if intent == "ask_philosophical_question": 
            current_depth = tracker.get_slot("philosophical_depth") or 1 
            events.append(SlotSet("philosophical_depth", current_depth + 1)) 

        #2. Se actualiza el nombre del humano si se ha mencionado
        if intent == "introduce_yourself":
            for entity in entities:
                if entity.get('entity') == 'personal_information' and entity.get('value') == 'nombre':
                    name_match = re.search(r"(?:me llamo|soy) (\w+)", user_message, re.IGNORECASE)
                    if name_match:
                        name_value = name_match.group(1)
                        events.append(SlotSet("human_name", name_value)) 
            
        return events 

class ActionRespondToGreeting(LlamaActionAdapter):

#Para detectar si es el primer saludo de la conversación o no

    def __init__(self):
        super().__init__(action_name="action_respond_to_greeting")

    def run(self, dispatcher, tracker, domain):
        is_first_greeting = tracker.get_slot("first_integration") or True
        events = super().run(dispatcher, tracker, domain)

        if is_first_greeting:
            events.append(SlotSet("first_integration", False))
            # Actualiza el slot para indicar que no es el primer saludo
        
        return events

class ActionRespondToIntroduction(LlamaActionAdapter):

#Para responder dependiendo de si ha encontrado una entidad a la que responder, y hacerlo en base a si el usuario ha dado su nombre o no (en el caso de que sí, guardarlo)
    def __init__(self):
        super().__init__(action_name="action_respond_to_introduction")

class ActionRespondToIdentityQuestion(LlamaActionAdapter):

#Información sobre su propia identidad (responde en diferentes niveles dependiendo de cuál sea la pregunta y cómo se haya formulado)
    def __init__(self):
        super().__init__(action_name="action_respond_to_identity_question")


class ActionRespondToEmotionQuestion(LlamaActionAdapter):

    def __init__(self):
        super().__init__(action_name="action_respond_to_emotion_question")

class ActionRespondToPhilosophicalQuestion(LlamaActionAdapter):

    def __init__(self):
        super().__init__(action_name="action_respond_to_philosophical_question")

class ActionRespondToHumanConceptExplanation(LlamaActionAdapter):

    def __init__(self):
        super().__init__(action_name="action_respond_to_concept_explanation")

class ActionRespondToBookQuestion(LlamaActionAdapter):

    def __init__(self):
        super().__init__(action_name="action_respond_to_book_question")

class ActionEndConversation(Action):
    """
    Acción final que proporciona un resumen de lo descubierto
    """
    
    def name(self) -> Text:
        return "action_end_conversation"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        discovered_info = tracker.get_slot("discovered_info") or []
        objective_manager = ObjectiveManager()
        final_status = objective_manager.check_completion(discovered_info)
        
        summary_message = f"""
        🎯 MISIÓN COMPLETADA
        
        Progreso total: {final_status['completion_percentage']:.1f}%
        Objetivos completados: {len(final_status['completed_objectives'])}/{len(objective_manager.objectives)}
        
        Has descubierto información valiosa sobre Sputnik. La conversación ha concluido exitosamente.
        """
        
        dispatcher.utter_message(text=summary_message)
        
        return [ConversationPaused()]

class ActionRespondToFallback(LlamaActionAdapter):
    
    def __init__(self):
        super().__init__(action_name="action_respond_to_fallback")
    
    def create_prompt(self, intent, entities, user_message, tracker):
        name = tracker.get_slot("human_name") or "Investigador"

        prompt = f"""Como Sputnik, estás hablando con {name}. No entiendes completamente lo que te está diciendo,
        pero debes responder de forma educada y curiosa. Puedes hacerle preguntas para clarificar.
        
        El mensaje exacto del humano es: '{user_message}'

        Debes responder mostrando interés pero admitiendo que necesitas más información o clarificación. Puedes incluir
        gestos sutiles entre asteriscos (*) que describan tu comportamiento físico.
        """

        return prompt

#class ActionRespondToTeaching(Action):



#class ActionRespondToPositiveEmotion(Action):



#class ActionRespondToNegativeEmotion(Action):



#class ActionRespondToCuriosity(Action):



#class ActionRespondToGoodbye(Action):



#class ActionRespondToFallback(Action):


#Acción para responder a un trigger que implique un cierre de conversación o un cambio de tema
#class ActionRespondtoTrigger(Action):